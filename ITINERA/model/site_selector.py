"""
智能选址推荐系统
基于ITINERA改造
"""

import os
import re
import copy
import json
import numpy as np
import concurrent.futures
import sys
import pandas as pd
import httpx

from model.utils.funcs import (
    RecurringList, compute_consecutive_distances, find_indices, 
    sample_items, reorder_list, remove_duplicates
)
from model.search import SearchEngine
from model.spatial import SpatialHandler


class DeepSeekClient:
    """轻量DeepSeek Chat Completions客户端，返回字符串content。"""
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = httpx.Client(timeout=30.0)

    def chat_json(self, messages: list, model: str = "deepseek-chat") -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"}
        }
        resp = self.session.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps({"rules": [], "synonyms": {}})


class SiteSelector:
    """
    智能选址推荐系统
    基于ITINERA的ItiNera类改造
    """
    
    def __init__(self, user_reqs, min_site_candidate_num=10, 
                 keep_prob=0.8, thresh=10000, proxy_call=None, 
                 city=None, type='zh', blend_w_text=0.5, blend_w_safe=0.5,
                 enable_safe=False, enable_llm_constraints=True, blend_w_struct=0.3,
                 deepseek_base_url=None, deepseek_api_key=None,
                 enable_spatial_optimization=False, enable_route_order=False,
                 min_distance_meters=0, dataset_path=None,
                 enable_struct_filters=False):
        
        # 核心参数
        self.MODEL = "gpt-4o"
        self.min_site_candidate_num = min_site_candidate_num
        
        # 处理用户需求
        self.type = type
        self.proxy = proxy_call
        self.user_reqs = user_reqs
        self.keep_prob = keep_prob
        self.thresh = thresh  # 空间聚类阈值（米）
        # 融合权重
        self.blend_w_text = float(blend_w_text)
        self.blend_w_safe = float(blend_w_safe)
        self.blend_w_struct = float(blend_w_struct)
        # 空间优化与访问顺序开关
        self.enable_spatial_optimization = bool(enable_spatial_optimization)
        self.enable_route_order = bool(enable_route_order)
        self.min_distance_meters = int(min_distance_meters) if min_distance_meters is not None else 0
        
        # 解析用户需求
        parsed_request = self.parse_user_request(user_reqs)
        self.parse_site_requirements(parsed_request)
        
        # 加载地块数据和embedding（支持自定义真实数据路径）
        self.load_site_data(city_name=city, dataset_path=dataset_path)
        
        # 初始化检索和空间处理模块
        self.maxSiteNum = 10  # 最多推荐10个地块
        self.search_engine = SearchEngine(
            embedding=self.embedding,
            emb_path=getattr(self, 'emb_path', ''),
            file_path=getattr(self, 'data_path', ''),
            proxy=self.proxy
        )
        self.spatial_handler = SpatialHandler(
            data=self.site_data,
            min_clusters=2,  # 至少2个空间聚类
            min_pois=self.maxSiteNum,
            citywalk=False,  # 选址不需要citywalk模式
            citywalk_thresh=self.thresh
        )
        
        # 初始化SAFE推理（按需启用）
        self.safe_enabled = False
        self.safe_pred_map = {}
        self.safe_config = None
        if enable_safe and self.blend_w_safe > 0:
            try:
                self.init_safe_inference()
            except Exception as e:
                # SAFE集成失败时不影响原有流程
                print(f"SAFE初始化失败：{e}")
                self.safe_enabled = False

        # 结构化过滤总开关（包含LLM规则与预设规则）；默认关闭，便于对比效果
        self.enable_struct_filters = bool(enable_struct_filters)

        # 初始化DeepSeek约束增强（按需启用）
        self.llm_constraints_enabled = False
        self.synonyms_map = {}
        self.deepseek_client = None
        try:
            if enable_llm_constraints and self.enable_struct_filters:
                # 优先使用显式传入的API Key，其次读取环境变量
                api_key = deepseek_api_key or os.environ.get('DEEPSEEK_API_KEY')
                base_url = deepseek_base_url or "https://api.deepseek.com"
                if api_key:
                    self.deepseek_client = DeepSeekClient(api_key=api_key, base_url=base_url)
                    self.llm_constraints_enabled = True
                else:
                    print("LLM约束未启用：缺少DEEPSEEK_API_KEY（未传入参数且环境变量不存在）")
        except Exception as e:
            print(f"DeepSeek初始化失败：{e}")
            self.llm_constraints_enabled = False

        # 文本分数缓存（用于前端展示与回退）
        self.text_score_map = {}
        self.text_score_min = 0.0
        self.text_score_max = 1.0

    def parse_user_request(self, user_reqs):
        """解析用户自然语言需求"""
        prompt = self.get_parse_prompt(user_reqs)
        response = self.proxy.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.MODEL
        ).replace("'", '"')
        
        try:
            return json.loads(response)
        except:
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print("解析JSON失败")
            return []

    # === 综合评分与权重推导 ===
    def _district_from_text(self, text: str | None) -> str | None:
        if not isinstance(text, str) or text.strip() == "":
            return None
        # 直接匹配常见行政区
        district_map = getattr(self, '_district_score_map', None)
        if isinstance(district_map, dict):
            for d in list(district_map.keys()):
                if d in text:
                    return d
        # 常见别名兜底
        aliases = {
            "广州天河": "天河区",
            "广州海珠": "海珠区",
            "广州越秀": "越秀区",
            "广州黄埔": "黄埔区",
            "广州荔湾": "荔湾区",
            "广州白云": "白云区",
            "广州番禺": "番禺区",
            "广州花都": "花都区",
            "广州南沙": "南沙区",
            "广州增城": "增城区",
            "广州从化": "从化区",
        }
        for k, v in aliases.items():
            if k in text:
                return v
        return None

    def _region_score(self, name_or_addr: str | None) -> float:
        # 默认行政区评分映射（可在__init__中覆盖）
        default_map = {
            "天河区": 9.5, "越秀区": 9.3, "海珠区": 9.0, "荔湾区": 8.5,
            "黄埔区": 7.8, "白云区": 7.5, "番禺区": 7.2, "花都区": 6.8,
            "南沙区": 6.5, "增城区": 6.2, "从化区": 6.0,
        }
        m = getattr(self, '_district_score_map', default_map)
        d = self._district_from_text(name_or_addr)
        if d and d in m:
            return float(m[d])
        return 7.0

    def _ensure_price_range(self):
        if getattr(self, '_price_min', None) is None or getattr(self, '_price_max', None) is None:
            try:
                s = pd.to_numeric(self.site_data["价格_万元/㎡"], errors='coerce')
                vmin = float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else 0.0
                vmax = float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else 1.0
                if abs(vmax - vmin) < 1e-8:
                    vmax = vmin + 1.0
                self._price_min, self._price_max = vmin, vmax
            except Exception:
                self._price_min, self._price_max = 0.0, 1.0

    def _price_score(self, price_val: float | None) -> float:
        """性价比分数：单价越低分数越高，归一化到[1,10]。"""
        self._ensure_price_range()
        try:
            v = float(price_val) if price_val is not None else np.nan
        except Exception:
            v = np.nan
        if np.isnan(v):
            return 5.0
        denom = (self._price_max - self._price_min)
        inv = (self._price_max - v) / denom
        return float(np.clip(1.0 + 9.0 * inv, 1.0, 10.0))

    def derive_scoring_weights(self) -> dict:
        """根据用户需求关键词/硬性约束推导权重。返回 {'traffic': w_a, 'price': w_b, 'region': w_c}"""
        w_a, w_b, w_c = 0.34, 0.33, 0.33
        # 正向需求关键词
        req_texts = []
        try:
            req_texts = [t for t in self.user_pos_reqs if isinstance(t, str)]
        except Exception:
            req_texts = [self.user_reqs] if isinstance(self.user_reqs, str) else []
        joined = " ".join(req_texts)
        # 交通偏好
        if any(k in joined for k in ["交通便利", "靠近地铁", "地铁", "公交", "交通"]):
            w_a = 0.5; w_b = 0.25; w_c = 0.25
        # 成本偏好
        if any(k in joined for k in ["性价比", "价格", "便宜", "预算", "成本"]):
            if w_a >= 0.5:
                w_a, w_b, w_c = 0.45, 0.45, 0.10
            else:
                w_a, w_b, w_c = 0.25, 0.5, 0.25
        # 硬性约束指定区域时弱化区域权重
        try:
            has_region_hard = any((c.get('type') == '区域' and not c.get('is_negative', False)) for c in getattr(self, 'hard_constraints', []))
        except Exception:
            has_region_hard = False
        if has_region_hard:
            w_c = min(w_c, 0.10)
            total = w_a + w_b + w_c
            if total > 1e-8:
                w_a, w_b, w_c = (w_a/total, w_b/total, w_c/total)
        return {"traffic": float(w_a), "price": float(w_b), "region": float(w_c)}

    def composite_score(self, site_id: int, weights: dict) -> float:
        """计算综合排序分数，范围[1,10]。"""
        try:
            row = self.site_data.loc[site_id]
        except Exception:
            return 5.0
        try:
            traffic_s = float(row.get('交通_便利评分(0-10)'))
        except Exception:
            traffic_s = 5.0
        traffic_s = float(np.clip(traffic_s, 0.0, 10.0))
        try:
            price_val = float(row.get('价格_万元/㎡'))
        except Exception:
            price_val = None
        price_s = self._price_score(price_val)
        try:
            addr = str(row.get('宗地坐落') or row.get('address') or row.get('name'))
        except Exception:
            addr = None
        region_s = self._region_score(addr)
        w_a = float(weights.get('traffic', 0.34))
        w_b = float(weights.get('price', 0.33))
        w_c = float(weights.get('region', 0.33))
        score = w_a * traffic_s + w_b * price_s + w_c * region_s
        return float(np.clip(score, 1.0, 10.0))

    def _intent_prioritize_traffic(self) -> bool:
        """根据用户需求文本判断是否明确强调交通便利。"""
        texts = []
        try:
            if isinstance(self.user_reqs, str):
                texts.append(self.user_reqs)
            if hasattr(self, 'user_pos_reqs') and isinstance(self.user_pos_reqs, list):
                texts.extend([t for t in self.user_pos_reqs if isinstance(t, str)])
        except Exception:
            pass
        all_text = ' '.join([str(t) for t in texts])
        keywords = ["交通", "交通便利", "便捷", "地铁", "公交", "通勤", "运输", "物流"]
        return any(k in all_text for k in keywords)

    def _intent_industrial(self) -> bool:
        """根据用户需求文本判断是否倾向工业/工厂用途。"""
        texts = []
        try:
            if isinstance(self.user_reqs, str):
                texts.append(self.user_reqs)
            if hasattr(self, 'user_pos_reqs') and isinstance(self.user_pos_reqs, list):
                texts.extend([t for t in self.user_pos_reqs if isinstance(t, str)])
        except Exception:
            pass
        all_text = ' '.join([str(t) for t in texts])
        keywords = ["工厂", "工业", "制造", "生产", "食品", "厂房", "产业园", "工业用地"]
        return any(k in all_text for k in keywords)

    def apply_request_overrides(self, sorted_results: np.ndarray) -> np.ndarray:
        """根据用户显式需求进行用途过滤与交通优先排序。
        - 若需求包含工业/工厂关键词：优先保留土地用途包含“工业”的候选；若过滤为空则回退。
        - 若需求强调交通便利：按列`交通_便利评分(0-10)`降序重排，并将第二列分数替换为该交通分。
        """
        try:
            if not isinstance(sorted_results, np.ndarray) or sorted_results.size == 0:
                return sorted_results

            columns = self.site_data.columns.tolist()
            filtered = sorted_results

            # 1) 用途过滤：工业
            if self._intent_industrial() and ('土地用途' in columns):
                try:
                    idxs = filtered[:, 0].astype(int)
                    series = self.site_data.loc[idxs, '土地用途'].astype(str)
                    mask = series.str.contains('工业', na=False)
                    keep = np.array(mask.values, dtype=bool)
                    after = filtered[keep]
                    if after.size > 0:
                        print(f"用途过滤(工业)：保留 {int(after.shape[0])}/{int(filtered.shape[0])}")
                        filtered = after
                    else:
                        print("用途过滤(工业)后为空，回退原结果")
                except Exception as e:
                    print(f"用途过滤失败：{e}")

            # 2) 交通优先排序
            if self._intent_prioritize_traffic() and ('交通_便利评分(0-10)' in columns):
                try:
                    idxs = filtered[:, 0].astype(int)
                    traffic = self.site_data.loc[idxs, '交通_便利评分(0-10)'].astype(float).clip(lower=0.0, upper=10.0)
                    fused = np.column_stack((idxs, traffic.values))
                    filtered = fused[fused[:, 1].argsort()[::-1]]
                    print("覆盖排序：按 交通_便利评分(0-10) 降序")
                except Exception as e:
                    print(f"交通优先排序失败：{e}")

            return filtered
        except Exception:
            return sorted_results

    def get_struct_constraint_prompt(self, constraints: list, columns: list, samples: dict):
        """构造结构化约束提示词，要求输出JSON：
        {
          "rules": [{"column": str, "op": str, "value": any, "negative": bool, "confidence": float}],
          "synonyms": {"原始约束文本": ["同义1", "同义2"]}
        }
        支持op: ==, in, contains, regex, <=, >=, <, >
        """
        try:
            examples = {col: samples.get(col, []) for col in columns}
        except Exception:
            examples = {}
        return (
            "你是城市选址约束工程师。请将以下硬性约束映射到数据表列的可执行规则，并给出同义/等价表达用于语义检索。\n"
            f"列名: {columns}\n示例值(部分): {json.dumps(examples, ensure_ascii=False)}\n"
            f"硬性约束: {json.dumps(constraints, ensure_ascii=False)}\n"
            "请返回JSON对象：{\n  \"rules\": [\n    {\"column\": str, \"op\": str, \"value\": any, \"negative\": bool, \"confidence\": float}\n  ],\n  \"synonyms\": {\"原始约束文本\": [\"同义1\", \"同义2\"]}\n}\n"
            "操作符仅限: ==, in, contains, regex, <=, >=, <, >。数值列使用数值比较，类别/文本列使用 contains/in/== 或 regex。"
        )

    def derive_pre_rules_from_hard_constraints(self, columns: list) -> list:
        """将常见需求直接映射为结构化列规则（无需LLM），用于增强/兜底。"""
        if not hasattr(self, 'hard_constraints') or len(self.hard_constraints) == 0:
            return []

        def col_exists(c):
            return c in columns

        def get_q(col: str, q: float, default: float) -> float:
            try:
                s = pd.to_numeric(self.site_data[col], errors='coerce')
                v = float(s.quantile(q))
                if np.isnan(v):
                    return default
                return v
            except Exception:
                return default

        price_low = get_q("价格_万元/㎡", 0.25, 0.0)
        price_high = get_q("价格_万元/㎡", 0.75, 999999.0)
        traffic_high = get_q("交通_便利评分(0-10)", 0.75, 7.5)

        def extract_distance(text: str) -> int | None:
            try:
                m = re.search(r"(\d{2,5})\s*米", text)
                if m:
                    return int(m.group(1))
            except Exception:
                pass
            return None

        def add_rule(lst, column, op, value, negative, confidence=0.9):
            if col_exists(column) and value is not None:
                lst.append({
                    "column": column,
                    "op": op,
                    "value": value,
                    "negative": bool(negative),
                    "confidence": float(confidence)
                })

        pre_rules = []

        subway_keys = ["地铁", "地铁站", "轨道", "metro", "subway"]
        bus_keys = ["公交", "公交站", "巴士", "bus"]
        train_keys = ["火车", "火车站", "铁路", "train"]
        park_keys = ["停车", "停车场", "parking"]
        traffic_keys = ["交通便利", "交通方便", "通勤便利", "出行便捷", "交通评分", "交通指数"]
        cheap_keys = ["便宜", "低价", "预算有限", "性价比", "价格便宜", "划算", "降成本"]
        expensive_keys = ["昂贵", "高端", "高价", "高档", "高预算"]

        for c in self.hard_constraints:
            txt = str(c.get("text", "")).lower()
            neg = bool(c.get("is_negative", False))
            if any(k in txt for k in subway_keys):
                add_rule(pre_rules, "交通_地铁数量(1.5km)", ">=", 1, neg)
                d = extract_distance(txt) or 800
                add_rule(pre_rules, "交通_地铁最近距离(m)", "<=", d, neg)
            if any(k in txt for k in bus_keys):
                add_rule(pre_rules, "交通_公交数量(0.5km)", ">=", 1, neg)
                d = extract_distance(txt) or 300
                add_rule(pre_rules, "交通_公交最近距离(m)", "<=", d, neg)
            if any(k in txt for k in train_keys):
                add_rule(pre_rules, "交通_火车数量(3km)", ">=", 1, neg)
                d = extract_distance(txt) or 2500
                add_rule(pre_rules, "交通_火车最近距离(m)", "<=", d, neg)
            if any(k in txt for k in park_keys):
                add_rule(pre_rules, "交通_停车数量(1km)", ">=", 1, neg)
                d = extract_distance(txt) or 800
                add_rule(pre_rules, "交通_停车最近距离(m)", "<=", d, neg)
            if any(k in txt for k in traffic_keys):
                add_rule(pre_rules, "交通_便利评分(0-10)", ">=", traffic_high, neg)
            if any(k in txt for k in cheap_keys):
                add_rule(pre_rules, "价格_万元/㎡", "<=", price_low, neg)
            if any(k in txt for k in expensive_keys):
                add_rule(pre_rules, "价格_万元/㎡", ">=", price_high, neg)

        return pre_rules

    def get_parse_prompt(self, user_input):
        """生成需求解析提示词"""
        return f"""
请分析用户的选址需求并拆解成结构化格式。

用户输入：{user_input}

返回JSON列表，每项包含：
- pos: 正向需求
- neg: 负向需求
- mustsee: 是否硬性约束(true/false)
- type: 类型（区域/用地类型/面积/成本/配套/其他）

示例输入："天河区20亩工业用地"
示例输出：
[
    {{"pos": "天河区", "neg": null, "mustsee": true, "type": "区域"}},
    {{"pos": "工业用地", "neg": null, "mustsee": true, "type": "用地类型"}},
    {{"pos": "20亩", "neg": null, "mustsee": false, "type": "面积"}}
]

示例输入："不超过500万，总价，尽量靠近地铁"
示例输出：
[
    {{"pos": "总价≤500万", "neg": null, "mustsee": true, "type": "成本"}},
    {{"pos": "靠近地铁", "neg": null, "mustsee": false, "type": "配套"}}
]

示例输入："不要工业用地，最好商业或办公"
示例输出：
[
    {{"pos": "商业用地", "neg": null, "mustsee": false, "type": "用地类型"}},
    {{"pos": "办公用地", "neg": null, "mustsee": false, "type": "用地类型"}},
    {{"pos": null, "neg": "工业用地", "mustsee": true, "type": "用地类型"}}
]

示例输入："至少30亩，临近学校和医院"
示例输出：
[
    {{"pos": "至少30亩", "neg": null, "mustsee": true, "type": "面积"}},
    {{"pos": "近学校", "neg": null, "mustsee": false, "type": "配套"}},
    {{"pos": "近医院", "neg": null, "mustsee": false, "type": "配套"}}
]

示例输入："浦东新区仓储用地，必须临近高速口"
示例输出：
[
    {{"pos": "浦东新区", "neg": null, "mustsee": true, "type": "区域"}},
    {{"pos": "仓储用地", "neg": null, "mustsee": true, "type": "用地类型"}},
    {{"pos": "临近高速出入口", "neg": null, "mustsee": true, "type": "配套"}}
]

示例输入："地价每平米不高于3000元，靠近港口或货运站"
示例输出：
[
    {{"pos": "地价≤3000元/㎡", "neg": null, "mustsee": true, "type": "成本"}},
    {{"pos": "近港口", "neg": null, "mustsee": false, "type": "配套"}},
    {{"pos": "近货运站", "neg": null, "mustsee": false, "type": "配套"}}
]

示例输入："不要噪声大的区域，远离化工园"
示例输出：
[
    {{"pos": null, "neg": "噪声大的区域", "mustsee": true, "type": "其他"}},
    {{"pos": null, "neg": "化工园", "mustsee": true, "type": "其他"}}
]

请严格按JSON格式返回，不要其他内容。
"""

    def parse_site_requirements(self, structured_input):
        """解析结构化需求"""
        # 将“must-see”从具体地块名称，改为“硬性约束”
        self.must_see_site_names = []  # 兼容旧字段，用于展示
        self.hard_constraints = []     # [{text, type, is_negative}]
        self.must_see_constraints_texts = []  # 仅用于提示词展示
        self.user_pos_reqs = []
        self.user_neg_reqs = []
        
        for req in structured_input:
            if req.get("mustsee") == True:
                # 将必须满足的需求作为硬性约束保存，而非名称匹配
                if req.get("pos"):
                    self.hard_constraints.append({
                        "text": req.get("pos"),
                        "type": req.get("type"),
                        "is_negative": False
                    })
                    self.must_see_constraints_texts.append(req.get("pos"))
                    self.must_see_site_names.append(req.get("pos"))  # 兼容旧提示词
                if req.get("neg"):
                    self.hard_constraints.append({
                        "text": req.get("neg"),
                        "type": req.get("type"),
                        "is_negative": True
                    })
                    self.must_see_constraints_texts.append(f"不包含:{req.get('neg')}")
                    self.must_see_site_names.append(f"不包含:{req.get('neg')}")  # 兼容旧提示词
            
            self.user_pos_reqs.append(req["pos"])
            if req.get("neg"):
                self.user_neg_reqs.append(req["neg"])
        
        # 如果没有正向需求，使用原始输入
        if len(self.user_pos_reqs) == 0:
            self.user_pos_reqs = [self.user_reqs]
            self.user_neg_reqs = [None]

    def load_site_data(self, city_name, dataset_path=None):
        """加载地块数据；支持自定义真实数据路径并标准化列。
        - 若提供 dataset_path（绝对或相对），优先使用；并把同名 .npy 作为embedding路径。
        - 否则回退到原来的 {city}_{type}.csv/.npy 命名。
        - 缺失的 name/address/desc 列会从可用列自动拼接生成。
        """
        import pandas as pd
        # 解析数据路径
        if dataset_path:
            data_path = dataset_path if os.path.isabs(dataset_path) else os.path.abspath(dataset_path)
            base, ext = os.path.splitext(data_path)
            emb_path = base + ".npy"
        else:
            data_path = os.path.join("model", "data", f'{city_name}_{self.type}.csv')
            emb_path = os.path.join("model", "data", f'{city_name}_{self.type}.npy')
        # 缓存路径，便于后续embedding维度不匹配时重算
        self.data_path = data_path
        self.emb_path = emb_path

        # 读取CSV数据
        self.site_data = pd.read_csv(data_path)
        # 标准化经纬度列
        if 'lon' not in self.site_data.columns and '经度' in self.site_data.columns:
            self.site_data = self.site_data.rename(columns={'经度': 'lon'})
        if 'lat' not in self.site_data.columns and '纬度' in self.site_data.columns:
            self.site_data = self.site_data.rename(columns={'纬度': 'lat'})
        # 标准化名称/地址/用途/面积/价格
        if 'name' not in self.site_data.columns:
            if '宗地坐落' in self.site_data.columns:
                self.site_data['name'] = self.site_data['宗地坐落'].astype(str)
            else:
                self.site_data['name'] = self.site_data.index.astype(str)
        if 'address' not in self.site_data.columns:
            if '宗地坐落' in self.site_data.columns:
                self.site_data['address'] = self.site_data['宗地坐落'].astype(str)
            else:
                self.site_data['address'] = self.site_data['name'].astype(str)
        # 生成desc/context（当源数据没有时）
        if 'desc' not in self.site_data.columns:
            usage = (self.site_data['土地用途'].astype(str) if '土地用途' in self.site_data.columns else pd.Series([''] * len(self.site_data)))
            area = (self.site_data['宗地面积(平方米)'].astype(str) if '宗地面积(平方米)' in self.site_data.columns else pd.Series([''] * len(self.site_data)))
            price = (self.site_data['挂牌起始价(万元)'].astype(str) if '挂牌起始价(万元)' in self.site_data.columns else pd.Series([''] * len(self.site_data)))
            self.site_data['desc'] = (
                ("用途:" + usage + "，面积:" + area + "㎡，起始价:" + price + "万元").str.strip()
            )
        if 'context' not in self.site_data.columns:
            self.site_data['context'] = (
                self.site_data['name'].astype(str) + "，地址是" + self.site_data['address'].astype(str) + "，" + self.site_data['desc'].astype(str)
            )
        # 填充ID
        if 'id' not in self.site_data.columns:
            self.site_data['id'] = self.site_data.index.astype(int)

        # 可选：生成平面坐标x/y（仅当后续启用空间优化时使用）
        if 'x' not in self.site_data.columns or 'y' not in self.site_data.columns:
            try:
                # 简化的近似换算（米）：
                # x ~ lon * 111320 * cos(lat)
                # y ~ lat * 110540
                rad = np.deg2rad(self.site_data['lat'].astype(float))
                self.site_data['x'] = self.site_data['lon'].astype(float) * 111320.0 * np.cos(rad)
                self.site_data['y'] = self.site_data['lat'].astype(float) * 110540.0
            except Exception:
                pass

        # 读取/生成embedding
        if os.path.exists(emb_path):
            self.embedding = np.load(emb_path)
        else:
            # 通过SearchEngine计算并保存embedding（支持缺省列的context拼接）
            se_tmp = SearchEngine(embedding=None, emb_path=emb_path, file_path=data_path, proxy=self.proxy)
            self.embedding = se_tmp.embedding
        
        # 初始化 must_see_sites 为索引列表（由约束过滤在候选检索阶段生成）
        self.must_see_sites = []
        
        # 创建索引映射
        self.site_data = self.site_data.reset_index(drop=True)
        row_idx = self.site_data.index.to_numpy()
        site_id = self.site_data["id"].to_numpy()
        self.r2i = {key: value for key, value in zip(row_idx, site_id)}
        self.i2r = {value: key for key, value in zip(row_idx, site_id)}

    def init_safe_inference(self):
        """加载SAFE配置与预测结果，并为站点计算geohash以便匹配。"""
        # 基于当前文件定位SAFE根目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        safe_home = os.path.join(base_dir, "SAFE", "SAFE")
        # 定位SAFE配置文件
        safe_cfg_path = os.path.join(safe_home, "config.json")
        if not os.path.exists(safe_cfg_path):
            # 未找到配置则禁用SAFE集成
            self.safe_enabled = False
            return
        with open(safe_cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.safe_config = cfg
        # 推断预测文件路径
        seed = cfg.get('infer_seed') or (cfg.get('seeds')[0] if cfg.get('seeds') else 0)
        # 如果未配置result_dir，则默认SAFE/out/results
        result_dir = cfg.get('result_dir', os.path.join(safe_home, 'out', 'results'))
        # 非绝对路径时，按SAFE根目录作为基准
        if not os.path.isabs(result_dir):
            result_dir = os.path.join(safe_home, result_dir)
        predictions_name = f"predictions_tab_only_seed_{seed}.csv"
        predictions_path = os.path.join(result_dir, predictions_name)
        # 读取预测CSV
        if os.path.exists(predictions_path):
            df = pd.read_csv(predictions_path)
            # 猜测列名
            geohash_col = next((c for c in df.columns if 'geohash' in c.lower()), None)
            proba_col = next((c for c in df.columns if c.lower() in ['proba1','prob_1','p1','proba','prob','prob1','prob_class_1']), None)
            if geohash_col and proba_col:
                self.safe_pred_map = dict(zip(df[geohash_col].astype(str), df[proba_col].astype(float)))
                self.safe_enabled = True
                print(f"[SAFE] 启用融合：加载到 {len(self.safe_pred_map)} 条预测，文件：{predictions_path}")
            else:
                self.safe_enabled = False
                print(f"[SAFE] 预测文件缺少必要列，禁用融合：{predictions_path}")
        else:
            self.safe_enabled = False
            print(f"[SAFE] 未找到预测文件，禁用融合：{predictions_path}")
        # 计算站点geohash
        precision = int(self.safe_config.get('geohash_precision', 12)) if self.safe_config else 12
        try:
            self.site_data['geohash'] = [self.encode_geohash(row['lat'], row['lon'], precision) for _, row in self.site_data[['lat','lon']].iterrows()]
        except Exception as e:
            print(f"计算站点geohash失败：{e}")
            self.safe_enabled = False

    def encode_geohash(self, lat, lon, precision=12):
        """使用SAFE的geohash实现编码经纬度。"""
        try:
            if not hasattr(self, '_safe_geohash_module'):
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                safe_root = os.path.join(base_dir, "SAFE", "SAFE")
                if safe_root not in sys.path:
                    sys.path.append(safe_root)
                import geohash as safe_geohash
                self._safe_geohash_module = safe_geohash
            return self._safe_geohash_module.encode(float(lat), float(lon), precision=precision)
        except Exception as e:
            raise e

    def blend_with_safe(self, sorted_results, w_text=None, w_safe=None):
        """将文本相似度与SAFE概率进行融合，返回新的排序结果。"""
        # w_safe<=0 或未启用SAFE时，直接返回文本排序
        if (w_safe is not None and float(w_safe) <= 0) or not self.safe_enabled or not isinstance(sorted_results, np.ndarray) or sorted_results.shape[1] < 2:
            return sorted_results
        # 使用实例默认权重（可覆盖）
        w_text = self.blend_w_text if w_text is None else float(w_text)
        w_safe = self.blend_w_safe if w_safe is None else float(w_safe)
        indices = sorted_results[:, 0].astype(int)
        text_scores = sorted_results[:, 1].astype(float)
        # 归一化文本分数到[0,1]
        min_s, max_s = float(text_scores.min()), float(text_scores.max())
        denom = (max_s - min_s) if (max_s - min_s) > 1e-8 else 1.0
        text_norm = (text_scores - min_s) / denom
        # SAFE概率
        try:
            geohashes = self.site_data.loc[indices, 'geohash'].astype(str).tolist()
            safe_probs = np.array([self.safe_pred_map.get(gh, np.nan) for gh in geohashes], dtype=float)
            na_mask = np.isnan(safe_probs)
            coverage = int((~na_mask).sum())
            total = int(len(safe_probs))
            if total > 0:
                print(f"[SAFE] 候选覆盖：{coverage}/{total}，融合权重(text={w_text}, safe={w_safe})")
            # 缺失填充
            if na_mask.any():
                median = np.nanmedian(safe_probs)
                if np.isnan(median):
                    median = 0.0
                safe_probs = np.where(na_mask, median, safe_probs)
            # 动态调整SAFE权重（按覆盖率缩放）
            try:
                coverage_ratio = (coverage / total) if total > 0 else 0.0
            except Exception:
                coverage_ratio = 0.0
            w_safe = w_safe * coverage_ratio
            self._last_safe_weight = float(w_safe)
        except Exception as e:
            print(f"[SAFE] 融合异常，回退文本排序：{e}")
            return sorted_results
        # 融合
        combined = w_text * text_norm + w_safe * safe_probs
        blended = np.column_stack((indices, combined))
        blended_sorted = blended[blended[:, 1].argsort()[::-1]]
        return blended_sorted

    def get_candidate_sites(self):
        """检索候选地块"""
        # 调试打印：用户需求拆解
        try:
            print("用户需求拆解：")
            print(f"- 正向需求: {self.user_pos_reqs}")
            print(f"- 负向需求: {self.user_neg_reqs}")
            if hasattr(self, 'hard_constraints'):
                hc_txt = [c.get('text') for c in self.hard_constraints]
                print(f"- 硬性约束文本: {hc_txt}")
        except Exception:
            pass
        
        def process_request(pos_req, neg_req):
            top_k = min(self.site_data.shape[0], self.min_site_candidate_num)
            req_sites = self.search_engine.query(
                desc=(pos_req, neg_req if neg_req else ""),
                top_k=top_k
            )
            # 打印每条子需求的检索结果（Top-K）
            try:
                ids = req_sites[:top_k, 0].astype(int).tolist() if req_sites is not None and len(req_sites) > 0 else []
                names = self.site_data.loc[ids, 'name'].astype(str).tolist() if ids else []
                print(f"子需求[{pos_req}] Top-{top_k} 地块: {names}")
            except Exception:
                pass
            pseudo_must_see_local = [int(site) for site in req_sites[:2, 0]]
            return req_sites, pseudo_must_see_local
        
        all_reqs_topk = []
        pseudo_must_see_sites = []
        
        # 并发处理多个需求
        if len(self.user_pos_reqs) > 1:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i, pos_req in enumerate(self.user_pos_reqs):
                    neg_req = self.user_neg_reqs[i] if i < len(self.user_neg_reqs) else None
                    future = executor.submit(process_request, pos_req, neg_req)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        req_sites, pseudo_must_see = future.result()
                        if req_sites is not None and len(req_sites) > 0:
                            pseudo_must_see_sites.extend(pseudo_must_see)
                            all_reqs_topk.append(req_sites)
                    except Exception as e:
                        print(f"处理需求时出错: {e}")
                        continue
        else:
            neg_req = self.user_neg_reqs[0] if self.user_neg_reqs else None
            try:
                req_sites, pseudo_must_see = process_request(self.user_pos_reqs[0], neg_req)
                if req_sites is not None and len(req_sites) > 0:
                    pseudo_must_see_sites.extend(pseudo_must_see)
                    all_reqs_topk.append(req_sites)
            except Exception as e:
                print(f"处理需求时出错: {e}")
        
        # 检查是否有有效结果
        if not all_reqs_topk:
            print("警告：没有找到任何候选地块")
            return np.empty((0, 2)), []
        
        # 合并结果
        all_reqs_topk = np.concatenate(all_reqs_topk, axis=0)
        unique_values = np.unique(all_reqs_topk[:, 0])
        result = [
            [value, all_reqs_topk[all_reqs_topk[:, 0] == value][:, 1].sum()]
            for value in unique_values
        ]
        result = np.array(result)
        # 打印合并后候选列表（ID与名称）
        try:
            merged_ids = result[:, 0].astype(int).tolist()
            merged_names = self.site_data.loc[merged_ids, 'name'].astype(str).tolist()
            print(f"合并后候选地块（{len(merged_ids)}个）: {merged_names}")
        except Exception:
            pass
        # 缓存文本分数（用于展示与回退）
        try:
            self.text_score_map = {int(v): float(s) for v, s in result.tolist()}
            self.text_score_min = float(result[:, 1].min()) if result.size else 0.0
            self.text_score_max = float(result[:, 1].max()) if result.size else 1.0
        except Exception:
            self.text_score_map = {}
            self.text_score_min = 0.0
            self.text_score_max = 1.0

        sorted_results = result[result[:, 1].argsort()[::-1]]
        
        # 先应用结构化约束过滤（若启用），减少候选规模并增强语义召回
        if self.enable_struct_filters:
            try:
                sorted_results = self.apply_struct_filters(sorted_results)
            except Exception as e:
                print(f"结构化约束过滤失败，回退原结果：{e}")
        else:
            try:
                print("结构化约束过滤已关闭")
            except Exception:
                pass

        # 基于需求的用途过滤与交通优先重排（不依赖LLM，直接数据驱动）
        try:
            sorted_results = self.apply_request_overrides(sorted_results)
        except Exception as e:
            try:
                print(f"需求覆盖失败，回退原结果：{e}")
            except Exception:
                pass

        # 与SAFE概率融合（暂时禁用）
        # try:
        #     sorted_results = self.blend_with_safe(sorted_results, w_text=self.blend_w_text, w_safe=self.blend_w_safe)
        # except Exception as e:
        #     print(f"SAFE融合失败：{e}")
        #     # 保持原有排序
        
        # 硬性约束文本召回（同义词增强）暂时禁用
        # try:
        #     sorted_results = self.apply_hard_constraints(sorted_results)
        # except Exception as e:
        #     print(f"约束过滤失败，回退原结果：{e}")
        
        return sorted_results, pseudo_must_see_sites

    def apply_struct_filters(self, sorted_results: np.ndarray) -> np.ndarray:
        """LLM增强的结构化过滤：
        - 使用DeepSeek将硬性约束映射为列级规则，并生成同义文本用于语义检索增强。
        - 计算结构化掩码并过滤候选，保留满足度更高的地块；保存同义词以供硬性约束文本过滤使用。
        - 计算结构化满足度(每站点满足的规则数/总规则数)，用于后续加权（可选）。
        """
        if not isinstance(sorted_results, np.ndarray) or sorted_results.size == 0:
            return sorted_results

        columns = self.site_data.columns.tolist()
        has_constraints = hasattr(self, 'hard_constraints') and len(self.hard_constraints) > 0
        pre_rules = self.derive_pre_rules_from_hard_constraints(columns) if has_constraints else []
        rules = []
        self.synonyms_map = {}

        if self.llm_constraints_enabled and has_constraints:
            # 为每列采样若干示例值，辅助映射
            samples = {}
            try:
                head = self.site_data.head(5)
                for col in columns:
                    vals = head[col].dropna().astype(str).tolist()
                    samples[col] = vals
            except Exception:
                samples = {}

            prompt = self.get_struct_constraint_prompt(constraints=self.hard_constraints, columns=columns, samples=samples)
            messages = [
                {"role": "system", "content": "你是专业的选址约束工程师，输出严格JSON"},
                {"role": "user", "content": prompt}
            ]
            try:
                resp = self.deepseek_client.chat_json(messages=messages, model="deepseek-chat")
                parsed = json.loads(resp)
                llm_rules = parsed.get("rules", []) or []
                self.synonyms_map = parsed.get("synonyms", {}) or {}
            except Exception as e:
                print(f"LLM结构化解析失败：{e}")
                llm_rules = []

            rules = pre_rules + llm_rules
            try:
                print(f"结构化约束：预设规则 {len(pre_rules)} 条，LLM规则 {len(llm_rules)} 条")
            except Exception:
                pass
        else:
            rules = pre_rules
            if len(pre_rules) > 0:
                print(f"结构化约束：使用预设规则 {len(pre_rules)} 条（未启用LLM或无硬性约束）")
            else:
                return sorted_results

        # 打印结构化约束解析摘要
        try:
            print(f"结构化约束：生成规则 {len(rules)} 条，同义词条目 {len(self.synonyms_map)}")
        except Exception:
            pass
        if len(rules) == 0:
            # 无规则则不做结构化过滤
            return sorted_results

        N = self.site_data.shape[0]
        # 初始化结构化满足度计数
        satisfied = np.zeros(N, dtype=float)
        total_rules = 0
        # 累积掩码，默认全保留
        struct_mask = np.ones(N, dtype=bool)

        def to_numeric_series(series: pd.Series) -> pd.Series:
            try:
                return pd.to_numeric(series, errors='coerce')
            except Exception:
                return pd.Series([np.nan] * len(series))

        for r in rules:
            col = r.get("column")
            op = (r.get("op") or "").lower()
            val = r.get("value")
            neg = bool(r.get("negative", False))
            conf = float(r.get("confidence", 0.0))
            if col not in columns or conf < 0.3:
                continue
            series = self.site_data[col]
            mask = np.ones(N, dtype=bool)
            try:
                if op in ["<=", ">=", "<", ">"]:
                    s_num = to_numeric_series(series)
                    v_num = None
                    try:
                        v_num = float(val)
                    except Exception:
                        v_num = np.nan
                    if np.isnan(v_num):
                        continue
                    if op == "<=":
                        mask = (s_num <= v_num)
                    elif op == ">=":
                        mask = (s_num >= v_num)
                    elif op == "<":
                        mask = (s_num < v_num)
                    elif op == ">":
                        mask = (s_num > v_num)
                elif op == "==":
                    mask = series.astype(str).str.lower() == str(val).lower()
                elif op == "contains":
                    mask = series.astype(str).str.contains(str(val), case=False, na=False)
                elif op == "regex":
                    try:
                        mask = series.astype(str).str.contains(str(val), flags=re.I, na=False)
                    except Exception:
                        mask = series.astype(str).str.contains(str(val), case=False, na=False)
                elif op == "in":
                    values = val if isinstance(val, list) else [val]
                    values = [str(v) for v in values if v is not None]
                    if len(values) == 0:
                        continue
                    comb = np.zeros(N, dtype=bool)
                    base = series.astype(str).str.lower()
                    for v in values:
                        comb = comb | base.str.contains(v.lower(), na=False)
                    mask = comb
                else:
                    # 未知操作符，跳过
                    continue
            except Exception:
                continue

            total_rules += 1
            if neg:
                struct_mask = struct_mask & (~mask)
                satisfied += (~mask).astype(float)
            else:
                struct_mask = struct_mask & mask
                satisfied += mask.astype(float)

        if total_rules > 0:
            struct_score = satisfied / float(total_rules)
        else:
            struct_score = np.ones(N, dtype=float)

        # 缓存结构化满足度，便于后续推荐解释
        try:
            self.struct_score_by_index = {int(i): float(struct_score[int(i)]) for i in range(N)}
        except Exception:
            self.struct_score_by_index = {}

        # 在候选排序上应用掩码
        keep_set = {int(i) for i in np.where(struct_mask)[0].tolist()}
        mask_res = np.array([int(i) in keep_set for i in sorted_results[:, 0]])
        filtered = sorted_results[mask_res]
        try:
            print(f"结构化约束过滤后数量：{int(filtered.shape[0])}/{int(sorted_results.shape[0])}")
        except Exception:
            pass

        if filtered.size == 0:
            print("结构化约束过滤后为空，回退原结果")
            return sorted_results

        # 排序仅使用文本分数（禁用 SAFE 与结构化满足度加权）
        try:
            idxs = filtered[:, 0].astype(int)
            t_scores = np.array([self.text_score_map.get(int(i), self.text_score_min) for i in idxs], dtype=float)
            denom = (self.text_score_max - self.text_score_min) if (self.text_score_max - self.text_score_min) > 1e-8 else 1.0
            t_norm = (t_scores - self.text_score_min) / denom
            fused = np.column_stack((idxs, t_norm))
            filtered = fused[fused[:, 1].argsort()[::-1]]
        except Exception:
            pass

        return filtered

    def apply_hard_constraints(self, sorted_results: np.ndarray) -> np.ndarray:
        """对候选结果应用硬性约束：
        - 正向约束：保留与约束文本相似度较高的前若干比例
        - 负向约束：排除与约束文本相似度较高的前若干比例
        最终返回过滤后的排序结果，并更新 self.must_see_sites 作为优化锚点。
        """
        if not hasattr(self, 'hard_constraints') or len(self.hard_constraints) == 0:
            # 无硬性约束，直接返回
            self.must_see_sites = []
            return sorted_results

        # 比例阈值（可调整）：取前35%作为“满足”或“需排除”的集合
        top_frac = 0.35
        N = self.site_data.shape[0]
        keep_set = set(range(N))
        anchor_sites = []

        for c in self.hard_constraints:
            text = c.get('text') or ''
            is_neg = c.get('is_negative', False)
            if not isinstance(text, str) or text.strip() == '':
                continue
            try:
                # 使用原始文本 + LLM同义词进行联合召回
                syns = []
                if hasattr(self, 'synonyms_map') and isinstance(self.synonyms_map, dict):
                    syns = self.synonyms_map.get(text, []) or []
                queries = [text] + [s for s in syns if isinstance(s, str) and s.strip() != '']
                union_top = set()
                for qtxt in queries:
                    q = self.search_engine.query(desc=(qtxt, ""), top_k=None)
                    if q.size == 0:
                        continue
                    k = max(1, int(len(q) * top_frac))
                    union_top.update([int(i) for i in q[:k, 0].tolist()])
                top_indices = list(union_top)

                if is_neg:
                    # 负向约束：从当前集合中剔除
                    keep_set = {i for i in keep_set if i not in set(top_indices)}
                else:
                    # 正向约束：与满足集合取交集，并记录一个锚点
                    keep_set = keep_set.intersection(set(top_indices))
                    # 选择一个锚点：对queries中第一个检索的top-1作为锚点（若存在）
                    try:
                        q0 = self.search_engine.query(desc=(queries[0], ""), top_k=None)
                        if q0.size > 0:
                            anchor_sites.append(int(q0[0, 0]))
                    except Exception:
                        pass
            except Exception as e:
                print(f"约束处理异常({text}): {e}")
                continue

        # 过滤排序结果
        if len(keep_set) == 0:
            print("警告：硬性约束过于严格，未找到满足的候选，回退未过滤结果")
            self.must_see_sites = []
            return sorted_results

        mask = np.array([int(i) in keep_set for i in sorted_results[:, 0]])
        filtered = sorted_results[mask]

        # 生成少量锚点（避免过多锚点导致聚类失败），最多取2个且需在filtered中
        anchors_unique = []
        for a in anchor_sites:
            if a in filtered[:, 0].astype(int).tolist() and a not in anchors_unique:
                anchors_unique.append(a)
            if len(anchors_unique) >= 2:
                break
        self.must_see_sites = anchors_unique

        if filtered.size == 0:
            print("约束过滤后为空，回退原结果")
            self.must_see_sites = []
            return sorted_results

        return filtered

    def optimize_site_selection(self, req_topk_sites, pseudo_must_see):
        """空间优化选址"""
        # 若关闭空间优化，采用简化策略：按分数排序取Top-K，并可选地做最小间距NMS，始终包含must_see
        if not self.enable_spatial_optimization:
            # 将输入转为(list of (id, score))
            pairs = [(int(i), float(s)) for i, s in req_topk_sites.tolist()]
            # 确保must_see在候选中
            for m in self.must_see_sites:
                if m not in [pid for pid, _ in pairs]:
                    pairs.insert(0, (int(m), 1000.0))
            # 按分数降序
            pairs.sort(key=lambda x: x[1], reverse=True)

            # 最小间距NMS，保证空间多样性（must_see不受限）
            def haversine(lon1, lat1, lon2, lat2):
                from math import radians, sin, cos, sqrt, atan2
                R = 6371000.0
                dlon = radians(lon2 - lon1)
                dlat = radians(lat2 - lat1)
                a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                return R * c

            selected_ids, selected_scores = [], []
            for pid, score in pairs:
                if pid in selected_ids:
                    continue
                allow = True
                if self.min_distance_meters and self.min_distance_meters > 0:
                    try:
                        row = self.site_data.loc[pid]
                        lon, lat = float(row['lon']), float(row['lat'])
                        for sid in selected_ids:
                            r2 = self.site_data.loc[sid]
                            d = haversine(lon, lat, float(r2['lon']), float(r2['lat']))
                            if d < self.min_distance_meters and pid not in self.must_see_sites:
                                allow = False
                                break
                    except Exception:
                        allow = True
                if allow:
                    selected_ids.append(pid)
                    selected_scores.append(score)
                if len(selected_ids) >= self.maxSiteNum:
                    break

            if not selected_ids:
                try:
                    if isinstance(req_topk_sites, np.ndarray) and req_topk_sites.ndim == 2 and req_topk_sites.shape[0] > 0:
                        selected_ids = [int(i) for i in req_topk_sites[:, 0].astype(int).tolist()[:self.maxSiteNum]]
                        selected_scores = [float(s) for s in req_topk_sites[:, 1].astype(float).tolist()[:self.maxSiteNum]]
                    else:
                        selected_ids, selected_scores = [], []
                except Exception:
                    selected_ids, selected_scores = [], []

            clusters = [selected_ids]
            return selected_ids, selected_scores, clusters

        # 默认分支：使用空间处理模块进行聚类优化
        site_ids = req_topk_sites[:, 0].astype(int).tolist()
        site_ids.extend(self.must_see_sites)
        all_site_ids = list(set(site_ids))

        sites, scores, clusters, _ = self.spatial_handler.get_poi_candidates(
            allpoi_idlist=all_site_ids,
            must_see_poi_idlist=self.must_see_sites,
            req_topk_pois=req_topk_sites,
            min_num_candidate=self.min_site_candidate_num,
            thresh=self.thresh,
            pseudo_must_see_pois=pseudo_must_see
        )

        if len(sites) > self.maxSiteNum:
            sites, scores, clusters = sample_items(
                sites, scores, clusters,
                keep_prob=self.maxSiteNum / len(sites),
                keep_ids=pseudo_must_see
            )
            clusters = [c for c in clusters if c]

        return sites, scores, clusters

    def generate_site_order(self, sites, clusters):
        """生成地块访问顺序"""
        # 关闭访问顺序生成时，保持当前排序并返回单一聚类
        if not self.enable_route_order:
            if clusters is None or len(clusters) == 0:
                clusters = [list(sites)] if isinstance(sites, (list, np.ndarray)) else [[]]
            return np.array(sites), list(range(len(clusters))), clusters

        # 调整聚类顺序
        order = reorder_list(sites, clusters)
        sites = np.array(sites)[order]
        
        # 计算聚类中心
        centroids = self.spatial_handler.get_cluster_centroids(clusters)
        
        # TSP求解聚类顺序
        clusters_order, _, _ = self.spatial_handler.get_tsp_order(
            locs=np.array(centroids)
        )
        
        # 调整起点
        recurring_order = list(clusters_order) + [clusters_order[0]]
        distances = compute_consecutive_distances(
            np.array(centroids), recurring_order
        )
        max_dist_idx = distances.argsort()[-1:][0]
        
        new_clusters_order = []
        new_clusters_order.extend(clusters_order[max_dist_idx + 1:])
        new_clusters_order.extend(clusters_order[:max_dist_idx + 1])
        
        return sites, new_clusters_order, clusters

    def generate_recommendation(self, ordered_sites, clusters):
        """生成推荐报告"""
        
        # 准备候选地块信息
        context_string = ""
        for i, site_id in enumerate(ordered_sites[:self.maxSiteNum]):
            site_info = self.site_data.loc[site_id]
            context = site_info['context'] if 'context' in site_info else site_info['desc']
            context_string += f'序号{i+1}: "{context[:100]}"\n'
        
        # 展示硬性约束文本（兼容旧字段）
        display_constraints = self.must_see_constraints_texts if hasattr(self, 'must_see_constraints_texts') else self.must_see_site_names
        must_see_string = str(display_constraints) if display_constraints else "无"
        
        # 生成提示词
        prompt = self.get_recommendation_prompt(
            context_string=context_string,
            must_see_string=must_see_string,
            keyword_reqs=self.user_pos_reqs,
            userReqList=self.user_reqs,
            maxSiteNum=min(self.maxSiteNum, len(ordered_sites)),
            numMustSee=len(self.must_see_sites),
            numCandidates=len(ordered_sites)
        )
        
        # 调用LLM
        messages = [
            {"role": "system", "content": "你是专业的选址顾问"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.proxy.chat(messages=messages, model=self.MODEL)
        
        try:
            result = json.loads(response)
        except:
            try:
                result = json.loads(response[8:-4])
            except:
                print("无法解析JSON响应")
                return {"error": response}
        
        # 归一化文本分数到[1,10]（用于展示与缺省回填）
        def norm_score(sid: int):
            s = self.text_score_map.get(int(sid))
            if s is None:
                return 5.0
            denom = (self.text_score_max - self.text_score_min)
            if denom <= 1e-8:
                return 5.0
            return 1.0 + 9.0 * ((s - self.text_score_min) / denom)

        # 添加坐标与GeoJSON，便于前端地图可视化，并按分数高低排序展示
        try:
            enriched_sites = {}
            features = []
            # 计算最终展示分：final = w_vector*text_norm + w_poi*poi_score（poi含综合分与规则分）
            display_ids = list(ordered_sites[:self.maxSiteNum])
            weights_poi = self.derive_scoring_weights()
            # 若明确强调交通便利，则在POI综合分中强化交通权重
            try:
                if self._intent_prioritize_traffic():
                    weights_poi = {'traffic': 0.80, 'price': 0.15, 'region': 0.05}
            except Exception:
                pass
            final_w_vector = float(self.blend_w_text)
            final_w_poi = float(1.0 - final_w_vector)
            poi_struct_ratio = 0.2  # POI内规则分占比，结构化满足度归一到[1,10]

            # 预先计算每个地块的最终分，便于排序
            score_by_id = {}
            poi_by_id = {}
            breakdown_by_id = {}
            for sid in display_ids:
                sid_int = int(sid)
                t_norm = norm_score(sid_int)
                comp_s = self.composite_score(sid_int, weights_poi)
                struct_raw = None
                if hasattr(self, 'struct_score_by_index') and isinstance(self.struct_score_by_index, dict):
                    struct_raw = self.struct_score_by_index.get(sid_int)
                struct_norm = (1.0 + 9.0 * float(struct_raw)) if (struct_raw is not None) else None
                if struct_norm is not None:
                    poi_s = float((1.0 - poi_struct_ratio) * comp_s + poi_struct_ratio * struct_norm)
                else:
                    poi_s = float(comp_s)
                final_s = final_w_vector * (t_norm if t_norm is not None else 5.0) + final_w_poi * poi_s
                final_s = float(np.clip(final_s, 1.0, 10.0))
                score_by_id[sid_int] = final_s
                poi_by_id[sid_int] = poi_s
                breakdown_by_id[sid_int] = {
                    'text_norm': t_norm,
                    'poi_composite': comp_s,
                    'struct_norm': struct_norm,
                    'poi_score': poi_s,
                    'final_score': final_s
                }
            # 按最终分排序（高到低）；若显式强调交通便利，则按交通分重排
            try:
                if self._intent_prioritize_traffic() and ('交通_便利评分(0-10)' in self.site_data.columns):
                    def traffic_s(sid):
                        try:
                            v = float(self.site_data.loc[int(sid), '交通_便利评分(0-10)'])
                            return float(np.clip(v, 0.0, 10.0))
                        except Exception:
                            return -float('inf')
                    display_ids.sort(key=lambda sid: traffic_s(sid), reverse=True)
                else:
                    display_ids.sort(key=lambda sid: score_by_id.get(int(sid), -float('inf')), reverse=True)
            except Exception:
                pass
            # 解释项准备
            debug_scores = {}

            for i, site_id in enumerate(display_ids):
                row = self.site_data.loc[site_id]
                key = str(i + 1)
                site_entry = result.get('sites', {}).get(key, {}) if isinstance(result.get('sites', {}), dict) else {}
                site_entry['id'] = str(row['id']) if 'id' in row else str(site_id)
                site_entry['lat'] = float(row['lat'])
                site_entry['lon'] = float(row['lon'])
                # 已禁用 SAFE 概率填充
                # 始终优先使用数据集中原始名称/宗地坐落，避免LLM产生的泛化名称
                try:
                    preferred_name = None
                    if 'name' in row and isinstance(row['name'], str) and row['name'].strip():
                        preferred_name = row['name'].strip()
                    elif '宗地坐落' in row and isinstance(row['宗地坐落'], str) and row['宗地坐落'].strip():
                        preferred_name = row['宗地坐落'].strip()
                    site_entry['name'] = preferred_name or site_entry.get('name') or f"地块{key}"
                except Exception:
                    site_entry['name'] = site_entry.get('name') or f"地块{key}"
                # 分数字段使用最终分（覆盖LLM分），确保展示逻辑一致
                try:
                    site_entry['score'] = float(score_by_id.get(int(site_id), norm_score(site_id)))
                except Exception:
                    try:
                        ns = norm_score(site_id)
                        site_entry['score'] = float(ns) if ns is not None else float('nan')
                    except Exception:
                        pass
                # 优势/风险回填，避免空展示
                if not site_entry.get('advantages'):
                    site_entry['advantages'] = row['context'] if ('context' in row and isinstance(row['context'], str)) else (row['desc'] if 'desc' in row else "")
                if not site_entry.get('risks'):
                    site_entry['risks'] = site_entry.get('risks') or ""
                if not site_entry.get('reason'):
                    site_entry['reason'] = (row['context'][:100] if ('context' in row and isinstance(row['context'], str)) else "")

                # 解释项：最终分拆解（向量/POI/规则）
                try:
                    bd = breakdown_by_id.get(int(site_id), {})
                    debug_scores[str(site_id)] = {
                        'text_norm': bd.get('text_norm'),
                        'poi_composite': bd.get('poi_composite'),
                        'struct_norm': bd.get('struct_norm'),
                        'poi_score': bd.get('poi_score'),
                        'final_score': bd.get('final_score'),
                        'w_vector': float(final_w_vector),
                        'w_poi': float(final_w_poi),
                        'poi_struct_ratio': float(poi_struct_ratio)
                    }
                except Exception:
                    pass

                enriched_sites[key] = site_entry
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [site_entry['lon'], site_entry['lat']]},
                    "properties": {
                        "index": i + 1,
                        "id": site_entry['id'],
                        "name": site_entry['name'],
                        "score": site_entry.get('score'),
                        "reason": site_entry.get('reason', "")
                    }
                })
            result['sites'] = enriched_sites
            result['debug_scores'] = debug_scores
            if features:
                lats = [f['geometry']['coordinates'][1] for f in features]
                lons = [f['geometry']['coordinates'][0] for f in features]
                center = [sum(lons) / len(lons), sum(lats) / len(lats)]
            else:
                center = [0.0, 0.0]
            result['geojson'] = {"type": "FeatureCollection", "features": features}
            result['features'] = features
            result['center'] = {"lon": center[0], "lat": center[1]}
        except Exception as _e:
            pass
        
        return result

    def get_recommendation_prompt(self, context_string, must_see_string, 
                                 keyword_reqs, userReqList, 
                                 maxSiteNum, numMustSee, numCandidates):
        """生成推荐提示词"""
        
        return f"""
你是专业的选址顾问。请根据候选地块和用户需求，推荐最优方案。

### 候选地块
{context_string}

### 用户需求
- 原始需求：{userReqList}
- 关键要求：{keyword_reqs}
- 必选条件：{must_see_string}

### 任务
从候选中选择最优的{maxSiteNum}个地块，生成推荐报告。

### 输出格式（严格JSON）
{{
    "recommendations": "地块1->地块2->地块3",
    "summary": "总体推荐理由",
    "sites": {{
        "1": {{
            "name": "地块名称",
            "reason": "推荐理由",
            "score": 8.5,
            "advantages": ["优势1", "优势2", "优势3"],
            "risks": ["风险1", "风险2"]
        }}
    }}
}}

### 评分标准
- 区位优势：交通、配套
- 成本因素：地价、开发成本
- 政策环境：用地性质、规划
- 发展潜力：未来增值空间

### 具体要求
- 每个地块的优势不少于3条、风险不少于2条；尽量引用上下文中的具体数据（如距离、评分、价格等）以增强可解释性。

请按JSON格式输出，每个地块评分1-10分。
"""

    def solve(self):
        """执行完整的选址推荐流程"""
        
        print("Step 1: 检索候选地块...")
        req_topk_sites, pseudo_must_see = self.get_candidate_sites()
        print(f"✓ 找到 {len(req_topk_sites)} 个候选地块")
        
        print("Step 2: 空间优化选址...")
        if not self.enable_spatial_optimization:
            print("✓ 按评分直接选取Top-K")
        sites, scores, clusters = self.optimize_site_selection(
            req_topk_sites, pseudo_must_see
        )
        print(f"✓ 保留 {len(sites)} 个地块")

        # —— 打印POI与规则算分（综合分拆解 + 结构化满足度 + 核心POI指标）——
        try:
            weights = self.derive_scoring_weights()
            print("\n[调试] 候选地块得分拆解（按保留顺序，最多展示前10个）：")
            header = (
                "序号 | id | 名称 | 交通分 | 性价比分 | 地区分 | 结构化分 | 综合分 | 价格(万元/㎡) | 地铁数(1.5km)/最近(m) | 公交数(0.5km)/最近(m) | 火车数(3km)/最近(m) | 停车数(1km)/最近(m)"
            )
            print(header)
            print("-" * len(header))
            max_show = int(min(10, len(sites)))
            for idx in range(max_show):
                sid = int(sites[idx])
                row = self.site_data.loc[sid]
                # 基础字段
                name = str(row.get('name') or row.get('宗地坐落') or f"地块{sid}")
                price_val = None
                try:
                    price_val = float(row.get('价格_万元/㎡'))
                except Exception:
                    price_val = None
                # 分数拆解
                try:
                    traffic_s = float(row.get('交通_便利评分(0-10)'))
                except Exception:
                    traffic_s = 5.0
                traffic_s = float(np.clip(traffic_s, 0.0, 10.0))
                price_s = self._price_score(price_val)
                addr = str(row.get('宗地坐落') or row.get('address') or row.get('name'))
                region_s = self._region_score(addr)
                comp_s = self.composite_score(sid, weights)
                struct_s = None
                try:
                    if hasattr(self, 'struct_score_by_index') and isinstance(self.struct_score_by_index, dict):
                        struct_s = float(self.struct_score_by_index.get(sid)) if sid in self.struct_score_by_index else None
                except Exception:
                    struct_s = None
                # POI核心指标
                def geti(col):
                    try:
                        v = row.get(col)
                        if v is None:
                            return None
                        return float(v)
                    except Exception:
                        return None
                poi_subway_cnt = geti('交通_地铁数量(1.5km)')
                poi_subway_near = geti('交通_地铁最近距离(m)')
                poi_bus_cnt = geti('交通_公交数量(0.5km)')
                poi_bus_near = geti('交通_公交最近距离(m)')
                poi_train_cnt = geti('交通_火车数量(3km)')
                poi_train_near = geti('交通_火车最近距离(m)')
                poi_park_cnt = geti('交通_停车数量(1km)')
                poi_park_near = geti('交通_停车最近距离(m)')

                print(
                    f"{idx+1:>2} | {sid} | {name} | "
                    f"{traffic_s:.2f} | {price_s:.2f} | {region_s:.2f} | "
                    f"{(struct_s if struct_s is not None else float('nan')):.2f} | {comp_s:.2f} | "
                    f"{(price_val if price_val is not None else float('nan')):.2f} | "
                    f"{(poi_subway_cnt if poi_subway_cnt is not None else float('nan')):.0f}/{(poi_subway_near if poi_subway_near is not None else float('nan')):.0f} | "
                    f"{(poi_bus_cnt if poi_bus_cnt is not None else float('nan')):.0f}/{(poi_bus_near if poi_bus_near is not None else float('nan')):.0f} | "
                    f"{(poi_train_cnt if poi_train_cnt is not None else float('nan')):.0f}/{(poi_train_near if poi_train_near is not None else float('nan')):.0f} | "
                    f"{(poi_park_cnt if poi_park_cnt is not None else float('nan')):.0f}/{(poi_park_near if poi_park_near is not None else float('nan')):.0f}"
                )
            # 打印权重摘要
            print("\n[调试] 当前权重: ", {k: float(v) for k, v in weights.items()})
        except Exception as e:
            try:
                print(f"[调试] 分解打印失败: {e}")
            except Exception:
                pass
        
        # 访问顺序生成（静默）
        ordered_sites, clusters_order, clusters = self.generate_site_order(
            sites, clusters
        )
        # 静默结束
        
        print("Step 3: 生成推荐报告...")
        recommendation = self.generate_recommendation(ordered_sites, clusters)
        
        print("\n" + "=" * 60)
        print("推荐结果：")
        print("=" * 60)
        print(json.dumps(recommendation, ensure_ascii=False, indent=2))
        
        return recommendation