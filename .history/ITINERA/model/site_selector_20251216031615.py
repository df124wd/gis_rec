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
                 city=None, type='zh', blend_w_text=0.5,
                 enable_llm_constraints=True, blend_w_struct=0.3,
                 deepseek_base_url=None, deepseek_api_key=None,
                 enable_spatial_optimization=False,
                 min_distance_meters=0, dataset_path=None,
                 enable_struct_filters=False,
                 enable_multi_objective=True,
                 top_k=5):
        
        # 核心参数
        # 模型名称由proxy自动选择（DeepSeek或OpenAI）
        self.MODEL = None  # None表示使用proxy的默认模型
        self.min_site_candidate_num = min_site_candidate_num
        
        # 处理用户需求
        self.type = type
        self.proxy = proxy_call
        self.user_reqs = user_reqs
        self.keep_prob = keep_prob
        self.thresh = thresh  # 空间聚类阈值（米）
        # 融合权重
        self.blend_w_text = float(blend_w_text)
        self.blend_w_struct = float(blend_w_struct)
        # 空间优化开关
        self.enable_spatial_optimization = bool(enable_spatial_optimization)
        self.min_distance_meters = int(min_distance_meters) if min_distance_meters is not None else 0
        # 多目标优化开关
        self.enable_multi_objective = bool(enable_multi_objective)
        
        # 解析用户需求
        parsed_request = self.parse_user_request(user_reqs)
        self.parse_site_requirements(parsed_request)
        
        # 加载地块数据和embedding（支持自定义真实数据路径）
        self.load_site_data(city_name=city, dataset_path=dataset_path)
        
        # 初始化检索和空间处理模块
        self.maxSiteNum = top_k  # 最终推荐数量（由参数控制）
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
        
        # SAFE功能已移除（过度工程化，效果不明显）

        # 结构化过滤功能已移除（过度工程化，与语义检索重叠）

        # 文本分数缓存（用于前端展示与回退）
        self.text_score_map = {}
        self.text_score_min = 0.0
        self.text_score_max = 1.0
        
        # 用LLM智能推导区位评分映射（根据用户需求动态生成）
        self._district_score_map = self._derive_region_scores_with_llm(user_reqs)
        
        # 用LLM智能推导理想面积范围（根据用户需求）
        self._ideal_area_range = self._derive_ideal_area_with_llm(user_reqs)

    def _derive_ideal_area_with_llm(self, user_reqs: str) -> dict:
        """用LLM根据用户需求推导理想面积范围"""
        
        prompt = f"""你是专业的选址顾问。请根据用户需求，推导理想的地块面积范围。

## 用户需求
{user_reqs}

## 任务
分析用户需求中的业务类型，推导适合的地块面积范围（平方米）。

## 参考标准
- 小型食品加工厂: 5000-15000㎡
- 中型食品生产厂: 15000-40000㎡
- 大型食品工厂: 40000-100000㎡
- 快递中转站: 3000-10000㎡
- 物流仓储中心: 20000-80000㎡
- 小型办公: 1000-5000㎡
- 商业零售: 2000-10000㎡

## 输出格式（严格JSON，不要markdown代码块）
{{
    "min_area": 10000,
    "max_area": 40000,
    "ideal_area": 20000,
    "reasoning": "食品生产厂需要中等规模场地"
}}
"""
        
        try:
            response = self.proxy.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL
            )
            
            # 增强JSON解析容错性
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # 尝试提取JSON部分（去除markdown代码块等）
                import re
                # 移除markdown代码块标记
                cleaned = re.sub(r'```json\s*', '', response)
                cleaned = re.sub(r'```\s*', '', cleaned)
                # 尝试提取JSON对象
                match = re.search(r'\{[\s\S]*\}', cleaned)
                if match:
                    result = json.loads(match.group())
                else:
                    raise ValueError("无法从响应中提取JSON")
            
            min_area = float(result.get('min_area', 5000))
            max_area = float(result.get('max_area', 50000))
            ideal_area = float(result.get('ideal_area', (min_area + max_area) / 2))
            reasoning = result.get('reasoning', '')
            
            if reasoning:
                print(f"[LLM面积推导] {reasoning}")
            print(f"[LLM面积范围] 理想: {ideal_area:.0f}㎡, 范围: {min_area:.0f}-{max_area:.0f}㎡")
            
            return {'min': min_area, 'max': max_area, 'ideal': ideal_area}
            
        except Exception as e:
            print(f"[LLM面积推导失败] {e}，使用默认范围")
            return {'min': 5000, 'max': 50000, 'ideal': 20000}

    def _derive_region_scores_with_llm(self, user_reqs: str) -> dict:
        """用LLM根据用户需求智能推导各行政区的区位评分"""
        
        prompt = f"""你是专业的选址顾问。请根据用户的选址需求，为广州市各行政区打分。

## 用户需求
{user_reqs}

## 广州市行政区列表
天河区、越秀区、海珠区、荔湾区、黄埔区、白云区、番禺区、花都区、南沙区、增城区、从化区

## 任务
根据用户需求的业务类型和偏好，为每个行政区打分（1-10分）。

## 评分原则
- 如果是商业、金融、总部办公类需求 → 天河、越秀等核心区高分
- 如果是工业、制造、仓储、物流类需求 → 花都、增城、南沙等产业区高分
- 如果是食品加工、快递中转站类需求 → 花都、白云、增城等交通便利的产业区高分
- 如果是住宅、教育配套类需求 → 天河、海珠、番禺等生活配套好的区高分
- 如果用户明确指定了某个区域 → 该区域满分10分

## 输出格式（严格JSON，不要其他内容）
{{
    "scores": {{
        "天河区": 7.0,
        "越秀区": 6.5,
        "海珠区": 6.0,
        "荔湾区": 5.5,
        "黄埔区": 8.0,
        "白云区": 7.5,
        "番禺区": 6.0,
        "花都区": 9.0,
        "南沙区": 8.5,
        "增城区": 8.0,
        "从化区": 7.0
    }},
    "reasoning": "简要说明评分理由（30字以内）"
}}

## 参考示例
- "花都区快递中转站" → 花都区=10.0, 白云区=8.5, 增城区=8.0, 天河区=5.0（物流产业适合郊区）
- "天河区写字楼办公" → 天河区=10.0, 越秀区=9.0, 海珠区=8.0, 花都区=4.0（商务办公适合核心区）
- "食品生产加工厂" → 花都区=9.5, 增城区=9.0, 从化区=8.5, 天河区=3.0（工业生产适合产业园区）

请分析用户需求并输出评分JSON。"""
        
        try:
            response = self.proxy.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL
            )
            
            result = json.loads(response)
            scores = result.get('scores', {})
            reasoning = result.get('reasoning', '')
            
            if reasoning:
                print(f"[LLM区位评分] {reasoning}")
            
            # 验证并补全
            default_scores = {
                "天河区": 7.0, "越秀区": 7.0, "海珠区": 7.0, "荔湾区": 7.0,
                "黄埔区": 7.0, "白云区": 7.0, "番禺区": 7.0, "花都区": 7.0,
                "南沙区": 7.0, "增城区": 7.0, "从化区": 7.0
            }
            for district in default_scores:
                if district not in scores or not isinstance(scores[district], (int, float)):
                    scores[district] = default_scores[district]
                else:
                    scores[district] = float(np.clip(scores[district], 1.0, 10.0))
            
            # 打印评分结果
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top3 = [f"{d}={s:.1f}" for d, s in sorted_scores[:3]]
            print(f"[LLM区位评分] Top3: {', '.join(top3)}")
            
            return scores
            
        except Exception as e:
            print(f"[LLM区位评分失败] {e}，使用默认评分")
            return {
                "天河区": 9.5, "越秀区": 9.3, "海珠区": 9.0, "荔湾区": 8.5,
                "黄埔区": 7.8, "白云区": 7.5, "番禺区": 7.2, "花都区": 6.8,
                "南沙区": 6.5, "增城区": 6.2, "从化区": 6.0
            }

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
        """根据LLM推导的动态区位评分映射计算分数"""
        # 使用LLM在初始化时生成的动态映射
        score_map = getattr(self, '_district_score_map', {})
        if not score_map:
            # 兜底默认值
            score_map = {
                "天河区": 7.0, "越秀区": 7.0, "海珠区": 7.0, "荔湾区": 7.0,
                "黄埔区": 7.0, "白云区": 7.0, "番禺区": 7.0, "花都区": 7.0,
                "南沙区": 7.0, "增城区": 7.0, "从化区": 7.0
            }
        
        district = self._district_from_text(name_or_addr)
        if district and district in score_map:
            return float(score_map[district])
        return 7.0  #

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
        """用LLM智能推导评价指标权重。返回 {'traffic': w_a, 'price': w_b, 'area': w_c, 'region': w_d}"""
        
        prompt = f"""你是专业的选址顾问。请根据用户需求，推导各评价指标的权重。

## 用户需求
{self.user_reqs}

## 可用评价指标
1. traffic - 交通便利性（地铁、公交、停车场、火车站的数量和距离）
2. price - 价格成本（地块单价，越低越好）
3. area - 地块规模（面积大小，通常越大越好）
4. region - 区位优势（所在行政区的发展水平，如天河>花都）

## 任务
根据用户需求的侧重点，为每个指标分配权重（0-1之间，总和为1）。

## 输出格式（严格JSON，不要其他内容）
{{
    "weights": {{
        "traffic": 0.30,
        "price": 0.25,
        "area": 0.20,
        "region": 0.25
    }},
    "reasoning": "简要说明权重分配理由（20字以内）"
}}

## 参考示例
- "花都区快递中转站，价格适中，交通便利" → traffic=0.35, price=0.30, area=0.15, region=0.20（物流需要交通便利和成本控制）
- "天河区20亩工业用地" → region=0.35, area=0.30, traffic=0.20, price=0.15（明确指定区域和面积）
- "便宜的大面积仓储用地" → price=0.40, area=0.35, traffic=0.15, region=0.10（强调价格和面积）
- "靠近地铁的商业用地" → traffic=0.50, region=0.25, price=0.15, area=0.10（强调交通便利）

请分析用户需求并输出权重JSON。"""
        
        try:
            response = self.proxy.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL
            )
            
            # 解析JSON
            result = json.loads(response)
            weights = result.get('weights', {})
            reasoning = result.get('reasoning', '')
            
            if reasoning:
                print(f"[LLM权重推导] {reasoning}")
            
            # 验证并补全缺失的权重
            default_weights = {"traffic": 0.30, "price": 0.25, "area": 0.20, "region": 0.25}
            for key in default_weights:
                if key not in weights or not isinstance(weights[key], (int, float)):
                    weights[key] = default_weights[key]
            
            # 归一化确保总和为1
            total = sum(weights.values())
            if total > 0:
                weights = {k: float(v)/total for k, v in weights.items()}
            
            print(f"[LLM权重结果] traffic={weights['traffic']:.2f}, price={weights['price']:.2f}, area={weights['area']:.2f}, region={weights['region']:.2f}")
            return weights
            
        except Exception as e:
            print(f"[LLM权重推导失败] {e}，使用默认权重")
            return {"traffic": 0.30, "price": 0.25, "area": 0.20, "region": 0.25}

    def composite_score(self, site_id: int, weights: dict) -> float:
        """计算综合排序分数，范围[1,10]。支持4个维度：交通、价格、面积、区位"""
        try:
            row = self.site_data.loc[site_id]
        except Exception:
            return 5.0
        
        # 1. 交通便利性评分（直接使用数据集中的评分）
        try:
            traffic_s = float(row.get('交通_便利评分(0-10)'))
        except Exception:
            traffic_s = 5.0
        traffic_s = float(np.clip(traffic_s, 0.0, 10.0))
        
        # 2. 价格评分（越低越好）
        try:
            price_val = float(row.get('价格_万元/㎡'))
        except Exception:
            price_val = None
        price_s = self._price_score(price_val)
        
        # 3. 面积评分（归一化到0-10）
        area_s = self._area_score(row)
        
        # 4. 区位评分
        try:
            addr = str(row.get('宗地坐落') or row.get('address') or row.get('name'))
        except Exception:
            addr = None
        region_s = self._region_score(addr)
        
        # 加权计算
        w_traffic = float(weights.get('traffic', 0.30))
        w_price = float(weights.get('price', 0.25))
        w_area = float(weights.get('area', 0.20))
        w_region = float(weights.get('region', 0.25))
        
        score = w_traffic * traffic_s + w_price * price_s + w_area * area_s + w_region * region_s
        return float(np.clip(score, 1.0, 10.0))
    
    def _area_score(self, row) -> float:
        """面积评分：基于LLM推导的理想面积范围计算匹配度"""
        try:
            area = float(row.get('宗地面积(平方米)', 0))
        except Exception:
            return 5.0
        
        # 获取LLM推导的理想面积范围
        ideal_range = getattr(self, '_ideal_area_range', {'min': 5000, 'max': 50000, 'ideal': 20000})
        min_area = ideal_range.get('min', 5000)
        max_area = ideal_range.get('max', 50000)
        ideal_area = ideal_range.get('ideal', (min_area + max_area) / 2)
        
        # 计算面积匹配度
        if min_area <= area <= max_area:
            # 在理想范围内，越接近理想值分数越高
            distance = abs(area - ideal_area)
            max_distance = max(ideal_area - min_area, max_area - ideal_area)
            if max_distance > 0:
                score = 10.0 - 3.0 * (distance / max_distance)  # 7-10分
            else:
                score = 10.0
        elif area < min_area:
            # 面积太小，按比例扣分
            ratio = area / min_area if min_area > 0 else 0
            score = 1.0 + 5.0 * ratio  # 1-6分
        else:
            # 面积太大，按比例扣分（但不会太低，大面积也有价值）
            excess_ratio = (area - max_area) / max_area if max_area > 0 else 1
            score = max(4.0, 7.0 - 3.0 * min(excess_ratio, 1.0))  # 4-7分
        
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

            # 用途过滤已移至两阶段多目标优化的硬约束过滤中

            # 交通优先排序
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

    # 结构化约束相关方法已移除（get_struct_constraint_prompt, derive_pre_rules_from_hard_constraints）
    # 原因：过度工程化，与语义检索重叠，LLM映射不稳定

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

    # SAFE相关方法已移除（init_safe_inference, encode_geohash, blend_with_safe）
    # 原因：过度工程化，效果不明显，已默认禁用

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
        
        # 结构化约束过滤已移除

        # 基于需求的用途过滤与交通优先重排（不依赖LLM，直接数据驱动）
        try:
            sorted_results = self.apply_request_overrides(sorted_results)
        except Exception as e:
            try:
                print(f"需求覆盖失败，回退原结果：{e}")
            except Exception:
                pass

        # SAFE融合已移除
        
        # 硬性约束文本召回（同义词增强）暂时禁用
        # try:
        #     sorted_results = self.apply_hard_constraints(sorted_results)
        # except Exception as e:
        #     print(f"约束过滤失败，回退原结果：{e}")
        
        return sorted_results, pseudo_must_see_sites

    # apply_struct_filters方法已移除（约170行代码）
    # 原因：过度工程化，与语义检索重叠，LLM映射不稳定

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
        """空间优化选址（集成多目标优化）"""
        # 若启用多目标优化，使用帕累托前沿
        if self.enable_multi_objective and not self.enable_spatial_optimization:
            return self._multi_objective_selection(req_topk_sites, pseudo_must_see)
        
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

    # generate_site_order方法已移除（TSP访问顺序）
    # 原因：选址推荐不需要访问顺序，与业务场景不符

    def _multi_objective_selection(self, req_topk_sites, pseudo_must_see):
        """
        NSGA-II 多目标优化选址（完整版）
        
        核心思路：
        - 一个解 = 一个方案 = 10个地块的组合
        - 遗传算法进化50代，种群100个方案
        - 输出帕累托前沿（多个最优权衡方案）
        - 根据用户偏好选择一个方案，再按总分排序取Top-5
        """
        from model.multi_objective import run_nsga2_optimization
        
        print("\n" + "="*80)
        print("[NSGA-II 多目标优化] 启动...")
        print("="*80)
        
        # ========== 获取语义检索候选地块 ==========
        candidate_ids = req_topk_sites[:, 0].astype(int).tolist()
        print(f"\n[语义检索候选] {len(candidate_ids)}个地块")
        
        # ========== 准备工作 ==========
        # 推导评价指标权重
        weights = self.derive_scoring_weights()
        print(f"\n[LLM推导权重]")
        print(f"  - 交通便利性(traffic): {weights.get('traffic', 0):.1%}")
        print(f"  - 价格成本(price): {weights.get('price', 0):.1%}")
        print(f"  - 地块规模(area): {weights.get('area', 0):.1%}")
        print(f"  - 区位优势(region): {weights.get('region', 0):.1%}")
        
        # 根据用户需求动态选择优化目标
        objectives = self._derive_objectives_v2()
        print(f"\n[优化目标] 共{len(objectives)}个目标:")
        for obj in objectives:
            direction = "↑越大越好" if obj['maximize'] else "↓越小越好"
            print(f"  - {obj.get('desc', obj['name'])}: {obj['name']} ({direction})")
        
        # ========== 智能解析硬约束 ==========
        executable_constraints = self._derive_executable_constraints()
        
        # ========== 提取候选地块数据 ==========
        candidate_data = self.site_data.loc[candidate_ids].copy().reset_index(drop=True)
        # 保存原始索引映射
        candidate_data['_original_idx'] = candidate_ids
        
        # 添加区位分数列（用于优化计算）
        candidate_data['region_score'] = candidate_data.apply(
            lambda row: self._region_score(str(row.get('宗地坐落', ''))), axis=1
        )
        
        # ========== 运行NSGA-II优化 ==========
        # 参数设置：
        # - n_select=10: 每个方案选择10个地块
        # - pop_size=100: 种群大小100
        # - n_generations=50: 进化50代
        # - top_k=5: 最终推荐5个地块
        
        result = run_nsga2_optimization(
            candidate_data=candidate_data,
            objectives=objectives,
            weights=weights,
            constraints=executable_constraints,
            n_select=min(10, len(candidate_data)),  # 每方案选10个地块
            pop_size=100,
            n_generations=50,
            top_k=self.maxSiteNum  # 最终推荐数量
        )
        
        # ========== 处理结果 ==========
        # 从推荐结果中提取原始索引和分数
        final_ids = []
        final_scores = []
        
        # 保存分数breakdown供generate_recommendation使用
        self._nsga2_score_breakdown = {}
        
        for site in result['recommended_sites']:
            # 获取原始索引
            original_idx = site.get('original_idx')
            if original_idx is not None:
                final_ids.append(original_idx)
                final_scores.append(site['total_score'])
                
                # 保存分数breakdown（直接使用score字段，已经是0-10分值）
                breakdown = site.get('breakdown', {})
                self._nsga2_score_breakdown[original_idx] = {
                    'traffic': breakdown.get('traffic', {}).get('score', 5.0),
                    'price': breakdown.get('price', {}).get('score', 5.0),
                    'area': breakdown.get('area', {}).get('score', 5.0),
                    'region': breakdown.get('region', {}).get('score', 5.0),
                    'final_score': site['total_score']
                }
        
        # 打印结果详情
        print(f"\n[推荐结果详情]")
        print("-"*90)
        print(f"{'排名':<4} {'名称':<35} {'交通':<6} {'价格':<8} {'面积':<10} {'总分':<6}")
        print("-"*90)
        
        for i, (idx, score) in enumerate(zip(final_ids, final_scores)):
            try:
                row = self.site_data.loc[idx]
                name = str(row.get('宗地坐落', ''))[:33]
                traffic = float(row.get('交通_便利评分(0-10)', 0))
                price = float(row.get('价格_万元/㎡', 0))
                area = float(row.get('宗地面积(平方米)', 0))
                print(f"{i+1:<4} {name:<35} {traffic:<6.1f} {price:<8.4f} {area:<10.0f} {score:<6.2f}")
            except Exception as e:
                print(f"{i+1:<4} [数据读取失败: {e}]")
        
        print("-"*90)
        
        # 打印帕累托前沿信息
        pareto_info = result.get('pareto_info', {})
        print(f"\n[帕累托前沿] {pareto_info.get('n_solutions', 0)}个非支配方案")
        print(f"[选中方案] 第{pareto_info.get('selected_idx', 0)}号方案，包含{len(result.get('selected_solution', {}).get('sites', []))}个地块")
        print(f"[统计] 候选:{result.get('original_count', 0)} → 过滤后:{result.get('filtered_count', 0)} → 最终推荐:{len(final_ids)}")
        print("="*80)
        
        clusters = [final_ids]
        return final_ids, final_scores, clusters
    
    def _derive_executable_constraints(self) -> list:
        """
        智能推导可执行的硬约束
        使用LLM将用户需求转换为数据库可执行的过滤条件
        """
        executable_constraints = []
        
        if not hasattr(self, 'hard_constraints') or not self.hard_constraints:
            return executable_constraints
        
        print(f"\n[硬约束智能解析]")
        print(f"  原始约束: {[c.get('text') for c in self.hard_constraints]}")
        
        for c in self.hard_constraints:
            text = c.get('text', '').strip()
            ctype = c.get('type', '')
            is_neg = c.get('is_negative', False)
            
            if not text:
                continue
            
            # 1. 区域约束：提取区域名称，模糊匹配宗地坐落
            if ctype == '区域':
                # 提取区域关键词
                districts = ['天河', '越秀', '海珠', '荔湾', '黄埔', 
                            '白云', '番禺', '花都', '南沙', '增城', '从化']
                for d in districts:
                    if d in text:
                        executable_constraints.append({
                            'field': '宗地坐落',
                            'operator': 'not_contains' if is_neg else 'contains',
                            'value': d,
                            'original_text': text
                        })
                        print(f"  → 区域约束: 宗地坐落 {'不包含' if is_neg else '包含'} '{d}'")
                        break
            
            # 2. 用地类型约束：用LLM推导对应的土地用途
            elif ctype == '用地类型' or any(k in text for k in ['厂', '工厂', '生产', '仓储', '物流', '商业', '办公', '居住']):
                # 根据用户描述推导土地用途
                land_type = self._infer_land_type(text)
                if land_type:
                    executable_constraints.append({
                        'field': '土地用途',
                        'operator': 'not_contains' if is_neg else 'contains',
                        'value': land_type,
                        'original_text': text
                    })
                    print(f"  → 用地约束: 土地用途 {'不包含' if is_neg else '包含'} '{land_type}' (推导自'{text}')")
            
            # 3. 面积约束
            elif ctype == '面积':
                import re
                number_match = re.search(r'[≥≤>=<]?\s*(\d+(?:\.\d+)?)', text)
                if number_match:
                    value = float(number_match.group(1))
                    # 单位转换
                    if '亩' in text:
                        value *= 666.67
                    elif '公顷' in text:
                        value *= 10000
                    
                    # 操作符
                    if '≥' in text or '>=' in text or '至少' in text or '不少于' in text or '以上' in text:
                        op = '>='
                    elif '≤' in text or '<=' in text or '不超过' in text or '以下' in text:
                        op = '<='
                    else:
                        op = '>='
                    
                    executable_constraints.append({
                        'field': '宗地面积(平方米)',
                        'operator': op,
                        'value': value,
                        'original_text': text
                    })
                    print(f"  → 面积约束: 宗地面积 {op} {value:.0f}㎡")
        
        print(f"  解析结果: {len(executable_constraints)}个可执行约束")
        return executable_constraints
    
    def _infer_land_type(self, text: str) -> str:
        """根据用户描述推导土地用途类型"""
        text_lower = text.lower()
        
        # 工业类
        if any(k in text_lower for k in ['工厂', '生产厂', '制造', '加工', '食品厂', '电子厂', '机械厂']):
            return '工业'
        if any(k in text_lower for k in ['仓储', '物流', '仓库', '配送']):
            return '工业'  # 仓储物流通常也是工业用地
        
        # 商业类
        if any(k in text_lower for k in ['商业', '商场', '购物', '零售', '店铺']):
            return '商业'
        if any(k in text_lower for k in ['办公', '写字楼', '总部']):
            return '商业'  # 办公通常是商业用地
        
        # 居住类
        if any(k in text_lower for k in ['住宅', '居住', '小区', '公寓']):
            return '居住'
        
        return ''
    
    def _find_original_index(self, site_row: pd.Series) -> int:
        """根据地块数据找到原始DataFrame中的索引"""
        try:
            # 用宗地坐落匹配
            name = site_row.get('宗地坐落', '')
            if name:
                matches = self.site_data[self.site_data['宗地坐落'] == name]
                if len(matches) > 0:
                    return matches.index[0]
            
            # 用坐标匹配
            lon = site_row.get('lon')
            lat = site_row.get('lat')
            if lon and lat:
                matches = self.site_data[
                    (self.site_data['lon'] == lon) & 
                    (self.site_data['lat'] == lat)
                ]
                if len(matches) > 0:
                    return matches.index[0]
        except:
            pass
        return None
    
    def _derive_objectives_v2(self):
        """
        根据用户需求动态推导优化目标（V2版本）
        返回带weight_key的目标列表，用于两阶段优化
        """
        # 分析用户需求文本
        req_texts = []
        try:
            if isinstance(self.user_reqs, str):
                req_texts.append(self.user_reqs)
            if hasattr(self, 'user_pos_reqs') and isinstance(self.user_pos_reqs, list):
                req_texts.extend([t for t in self.user_pos_reqs if isinstance(t, str)])
        except Exception:
            pass
        all_text = ' '.join([str(t) for t in req_texts]).lower()
        
        # 判断是否强调性价比（单价）
        use_unit_price = any(k in all_text for k in ["性价比", "单价", "每平米", "每平方米", "元/㎡"])
        
        # 判断是否是工业/物流类需求
        is_industrial = any(k in all_text for k in ["工业", "工厂", "物流", "仓储", "生产", "制造"])
        
        # 基础4个目标
        objectives = [
            {
                'name': '交通_便利评分(0-10)', 
                'maximize': True, 
                'weight_key': 'traffic',
                'desc': '交通便利性'
            },
            {
                'name': '价格_万元/㎡' if use_unit_price else '挂牌起始价(万元)', 
                'maximize': False, 
                'weight_key': 'price',
                'desc': '价格成本（越低越好）'
            },
            {
                'name': '宗地面积(平方米)', 
                'maximize': True, 
                'weight_key': 'area',
                'desc': '地块规模'
            },
            {
                'name': 'region_score', 
                'maximize': True, 
                'weight_key': 'region',
                'desc': '区位优势'
            }
        ]
        
        return objectives
    
    def _map_weights_to_objectives(self, weights, objectives):
        """
        将评价指标权重映射到优化目标（4个目标）
        """
        obj_weights = {}
        for obj in objectives:
            name = obj['name']
            if '交通' in name:
                obj_weights[name] = weights.get('traffic', 0.25)
            elif '价格' in name or '起始价' in name:
                obj_weights[name] = weights.get('price', 0.25)
            elif '面积' in name:
                obj_weights[name] = weights.get('area', 0.25)
            elif 'region' in name or '区位' in name:
                obj_weights[name] = weights.get('region', 0.25)
            else:
                obj_weights[name] = 0.1
        
        # 归一化
        total = sum(obj_weights.values())
        if total > 0:
            obj_weights = {k: v/total for k, v in obj_weights.items()}
        
        return obj_weights
    
    def _generate_site_analysis(self, name: str, context: str, land_use: str, 
                                total_price: float, area: float, scores: dict, 
                                user_reqs: str) -> dict:
        """用LLM为单个地块生成详细的优势/风险分析"""
        
        # 获取理想面积范围
        ideal_range = getattr(self, '_ideal_area_range', {'min': 5000, 'max': 50000, 'ideal': 20000})
        
        prompt = f"""你是专业的选址顾问，请为以下地块生成详细的优势和风险分析。

## 地块基本信息
- 名称/位置: {name}
- 土地用途: {land_use}
- 面积: {area:.0f}平方米（约{area/666.67:.1f}亩）
- 总价: {total_price:.0f}万元
- 单价: {total_price/area*10000:.0f}元/㎡

## 地块详细描述
{context}

## 系统评分（满分10分）
- 交通便利性: {scores.get('traffic', 5):.2f}分
- 性价比: {scores.get('price', 5):.2f}分（单价越低分越高）
- 面积匹配度: {scores.get('area', 5):.2f}分（理想范围{ideal_range['min']:.0f}-{ideal_range['max']:.0f}㎡）
- 区位优势: {scores.get('region', 5):.2f}分
- 综合评分: {scores.get('total', 5):.2f}分

## 用户需求
{user_reqs}

## 分析任务
请从以下角度全面分析该地块：

1. **优势分析**（3-4条）：
   - 评分高的指标（≥7分）要重点说明
   - 结合用户需求分析适配度
   - 考虑该区域的产业环境、政策优势
   - 分析土地用途与用户业务的匹配度

2. **风险分析**（2-3条）：
   - 评分低的指标（<5分）要指出具体问题
   - 分析可能的隐性成本（配套建设、交通不便等）
   - 考虑未来发展的限制因素
   - 提出需要实地考察确认的事项

## 输出格式（严格JSON）
{{
    "advantages": [
        "优势1：具体描述（引用评分数据）",
        "优势2：具体描述",
        "优势3：具体描述",
        "优势4：具体描述"
    ],
    "risks": [
        "风险1：具体描述（引用评分数据）",
        "风险2：具体描述",
        "风险3：具体描述"
    ]
}}

要求：
1. 每条优势/风险要具体、有数据支撑
2. 内容要与用户需求紧密相关
3. 每条40-60字，信息量充足"""
        
        try:
            response = self.proxy.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL
            )
            
            # 尝试解析JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # 尝试提取JSON部分
                import re
                match = re.search(r'\{[\s\S]*\}', response)
                if match:
                    result = json.loads(match.group())
                else:
                    raise ValueError("无法解析LLM响应")
            
            advantages = result.get('advantages', [])
            risks = result.get('risks', [])
            
            # 确保返回的是列表
            if not isinstance(advantages, list):
                advantages = [str(advantages)] if advantages else []
            if not isinstance(risks, list):
                risks = [str(risks)] if risks else []
            
            return {
                'advantages': advantages,
                'risks': risks
            }
        except Exception as e:
            print(f"[LLM分析异常] {name[:20]}: {e}")
            raise e
    
    def _apply_spatial_diversity(self, site_ids):
        """
        应用空间多样性过滤（最小间距NMS）
        """
        def haversine(lon1, lat1, lon2, lat2):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371000.0
            dlon = radians(lon2 - lon1)
            dlat = radians(lat2 - lat1)
            a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        
        selected = []
        for sid in site_ids:
            if sid in self.must_see_sites:
                selected.append(sid)
                continue
            
            allow = True
            try:
                row = self.site_data.loc[sid]
                lon, lat = float(row['lon']), float(row['lat'])
                for s in selected:
                    r2 = self.site_data.loc[s]
                    d = haversine(lon, lat, float(r2['lon']), float(r2['lat']))
                    if d < self.min_distance_meters:
                        allow = False
                        break
            except Exception:
                allow = True
            
            if allow:
                selected.append(sid)
        
        return selected

    def generate_recommendation(self, ordered_sites, clusters):
        """生成推荐报告"""
        
        # 准备候选地块信息（包含完整的评分数据）
        context_string = ""
        for i, site_id in enumerate(ordered_sites[:self.maxSiteNum]):
            site_info = self.site_data.loc[site_id]
            
            # 获取各项评分数据
            traffic_score = float(site_info.get('交通_便利评分(0-10)', 0))
            price = float(site_info.get('挂牌起始价(万元)', 0))
            area = float(site_info.get('宗地面积(平方米)', 0))
            addr = str(site_info.get('宗地坐落', ''))
            region_score = self._region_score(addr)
            land_use = str(site_info.get('土地用途', ''))
            
            # 交通详情
            subway_count = int(site_info.get('交通_地铁数量(1.5km)', 0))
            bus_count = int(site_info.get('交通_公交数量(0.5km)', 0))
            parking_count = int(site_info.get('交通_停车数量(1km)', 0))
            
            # 构建详细的context
            context_string += f"""序号{i+1}: {addr}
  - 土地用途: {land_use}
  - 面积: {area:.0f}平方米，起始价: {price:.0f}万元
  - 交通便利评分: {traffic_score:.2f}分（1.5km内地铁{subway_count}个，500m内公交{bus_count}个，1km内停车场{parking_count}个）
  - 区位评分: {region_score:.1f}分
"""
        
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
                    weights_poi = {'traffic': 0.50, 'price': 0.20, 'area': 0.15, 'region': 0.15}
            except Exception:
                pass
            
            # 确保权重包含所有4个键
            default_weights = {'traffic': 0.25, 'price': 0.25, 'area': 0.25, 'region': 0.25}
            for key in default_weights:
                if key not in weights_poi:
                    weights_poi[key] = default_weights[key]
            
            # 预先计算每个地块的最终分
            # 优先使用NSGA-II返回的分数（如果有），否则重新计算
            score_by_id = {}
            breakdown_by_id = {}
            nsga2_breakdown = getattr(self, '_nsga2_score_breakdown', {})
            
            for sid in display_ids:
                sid_int = int(sid)
                
                # 优先使用NSGA-II的分数
                if sid_int in nsga2_breakdown:
                    bd = nsga2_breakdown[sid_int]
                    final_s = bd.get('final_score', 5.0)
                    traffic_s = bd.get('traffic', 5.0)
                    price_s = bd.get('price', 5.0)
                    area_s = bd.get('area', 5.0)
                    region_s = bd.get('region', 5.0)
                else:
                    # 回退：重新计算
                    try:
                        row = self.site_data.loc[sid_int]
                        
                        # 获取4个指标的归一化分数（都是0-10分）
                        traffic_s = float(row.get('交通_便利评分(0-10)', 5.0))
                        traffic_s = float(np.clip(traffic_s, 0.0, 10.0))
                        
                        price_s = self._price_score(row.get('价格_万元/㎡'))
                        area_s = self._area_score(row)
                        
                        addr = str(row.get('宗地坐落', ''))
                        region_s = self._region_score(addr)
                        
                        # 加权求和
                        final_s = (
                            weights_poi.get('traffic', 0.25) * traffic_s +
                            weights_poi.get('price', 0.25) * price_s +
                            weights_poi.get('area', 0.25) * area_s +
                            weights_poi.get('region', 0.25) * region_s
                        )
                        final_s = float(np.clip(final_s, 1.0, 10.0))
                        
                    except Exception:
                        final_s = 5.0
                        traffic_s = price_s = area_s = region_s = 5.0
                
                score_by_id[sid_int] = final_s
                breakdown_by_id[sid_int] = {
                    'traffic': traffic_s,
                    'price': price_s,
                    'area': area_s,
                    'region': region_s,
                    'final_score': final_s
                }
            
            # 不再重新排序，保持NSGA-II返回的顺序（已按总分排序）
            # 如果没有使用NSGA-II，则按分数排序
            if not nsga2_breakdown:
                try:
                    if self._intent_prioritize_traffic() and ('交通_便利评分(0-10)' in self.site_data.columns):
                        def traffic_s_func(sid):
                            try:
                                v = float(self.site_data.loc[int(sid), '交通_便利评分(0-10)'])
                                return float(np.clip(v, 0.0, 10.0))
                            except Exception:
                                return -float('inf')
                        display_ids.sort(key=lambda sid: traffic_s_func(sid), reverse=True)
                    else:
                        display_ids.sort(key=lambda sid: score_by_id.get(int(sid), -float('inf')), reverse=True)
                except Exception:
                    pass
            # 解释项准备
            debug_scores = {}
            
            # 先收集所有地块的基本信息，用于并发LLM调用
            site_info_list = []
            for i, site_id in enumerate(display_ids):
                row = self.site_data.loc[site_id]
                key = str(i + 1)
                site_entry = result.get('sites', {}).get(key, {}) if isinstance(result.get('sites', {}), dict) else {}
                site_entry['id'] = str(row['id']) if 'id' in row else str(site_id)
                site_entry['lat'] = float(row['lat'])
                site_entry['lon'] = float(row['lon'])
                site_entry['site_index'] = int(site_id)  # DataFrame行索引，用于获取POI详情
                
                # 获取名称
                try:
                    preferred_name = None
                    if 'name' in row and isinstance(row['name'], str) and row['name'].strip():
                        preferred_name = row['name'].strip()
                    elif '宗地坐落' in row and isinstance(row['宗地坐落'], str) and row['宗地坐落'].strip():
                        preferred_name = row['宗地坐落'].strip()
                    site_entry['name'] = preferred_name or site_entry.get('name') or f"地块{key}"
                except Exception:
                    site_entry['name'] = site_entry.get('name') or f"地块{key}"
                
                # 分数
                try:
                    site_entry['score'] = float(score_by_id.get(int(site_id), norm_score(site_id)))
                except Exception:
                    try:
                        ns = norm_score(site_id)
                        site_entry['score'] = float(ns) if ns is not None else float('nan')
                    except Exception:
                        pass
                
                # 获取评分数据
                bd = breakdown_by_id.get(int(site_id), {})
                traffic_s = bd.get('traffic', 5.0)
                price_s = bd.get('price', 5.0)
                area_s = bd.get('area', 5.0)
                region_s = bd.get('region', 5.0)
                final_s = bd.get('final_score', 5.0)
                
                # 获取地块详细信息
                context_text = row.get('context', '') if 'context' in row else ''
                land_use = str(row.get('土地用途', ''))
                total_price = float(row.get('挂牌起始价(万元)', 0))
                area_raw = float(row.get('宗地面积(平方米)', 0))
                
                site_entry['reason'] = f"综合评分{final_s:.2f}分"
                
                # 解释项
                try:
                    debug_scores[str(site_id)] = {
                        'traffic': bd.get('traffic'),
                        'price': bd.get('price'),
                        'area': bd.get('area'),
                        'region': bd.get('region'),
                        'final_score': bd.get('final_score'),
                        'weights': weights_poi
                    }
                except Exception:
                    pass
                
                # 收集信息用于并发LLM调用
                site_info_list.append({
                    'index': i,
                    'key': key,
                    'site_id': site_id,
                    'site_entry': site_entry,
                    'row': row,
                    'name': site_entry.get('name', ''),
                    'context': context_text,
                    'land_use': land_use,
                    'total_price': total_price,
                    'area': area_raw,
                    'scores': {'traffic': traffic_s, 'price': price_s, 'area': area_s, 'region': region_s, 'total': final_s}
                })
            
            # 并发调用LLM生成优势/风险分析
            print(f"[并发LLM] 开始为{len(site_info_list)}个地块生成优势/风险分析...")
            
            def generate_analysis_for_site(info):
                """单个地块的LLM分析任务"""
                try:
                    llm_result = self._generate_site_analysis(
                        name=info['name'],
                        context=info['context'],
                        land_use=info['land_use'],
                        total_price=info['total_price'],
                        area=info['area'],
                        scores=info['scores'],
                        user_reqs=self.user_reqs
                    )
                    return {
                        'index': info['index'],
                        'advantages': llm_result.get('advantages', [f"综合评分{info['scores']['total']:.2f}分"]),
                        'risks': llm_result.get('risks', ["建议实地考察确认"]),
                        'success': True
                    }
                except Exception as e:
                    # 回退到规则生成
                    scores = info['scores']
                    advantages = []
                    if scores['traffic'] >= 6: advantages.append(f"交通便利（{scores['traffic']:.2f}分）")
                    if scores['price'] >= 7: advantages.append(f"性价比高（{scores['price']:.2f}分）")
                    if scores['area'] >= 6: advantages.append(f"面积适中（{scores['area']:.2f}分）")
                    if scores['region'] >= 8: advantages.append(f"区位优越（{scores['region']:.2f}分）")
                    
                    risks = []
                    if scores['traffic'] < 4: risks.append(f"交通便利性较低（{scores['traffic']:.2f}分）")
                    if scores['price'] < 4: risks.append(f"性价比较低（{scores['price']:.2f}分）")
                    if scores['area'] < 4: risks.append(f"面积偏小（{scores['area']:.2f}分）")
                    if scores['region'] < 6: risks.append(f"区位一般（{scores['region']:.2f}分）")
                    
                    return {
                        'index': info['index'],
                        'advantages': advantages if advantages else ["综合表现均衡"],
                        'risks': risks if risks else ["建议实地考察确认"],
                        'success': False,
                        'error': str(e)
                    }
            
            # 使用线程池并发执行
            llm_results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_info = {executor.submit(generate_analysis_for_site, info): info for info in site_info_list}
                for future in concurrent.futures.as_completed(future_to_info):
                    try:
                        res = future.result()
                        llm_results[res['index']] = res
                        status = "✓" if res.get('success') else "✗"
                        print(f"  [{status}] 地块{res['index']+1} 分析完成")
                    except Exception as e:
                        info = future_to_info[future]
                        print(f"  [✗] 地块{info['index']+1} 分析异常: {e}")
            
            print(f"[并发LLM] 完成，成功{sum(1 for r in llm_results.values() if r.get('success'))}个")
            
            # 组装最终结果
            for info in site_info_list:
                site_entry = info['site_entry']
                llm_res = llm_results.get(info['index'], {})
                site_entry['advantages'] = llm_res.get('advantages', ["综合表现均衡"])
                site_entry['risks'] = llm_res.get('risks', ["建议实地考察确认"])
                
                enriched_sites[info['key']] = site_entry
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [site_entry['lon'], site_entry['lat']]},
                    "properties": {
                        "index": info['index'] + 1,
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
        
        # 添加子需求和权重信息供前端展示
        result['parsed_requirements'] = {
            'original': self.user_reqs,
            'positive': self.user_pos_reqs,
            'negative': self.user_neg_reqs,
            'hard_constraints': [c.get('text', '') for c in getattr(self, 'hard_constraints', [])]
        }
        
        # 添加LLM推导的权重
        try:
            weights = self.derive_scoring_weights()
            result['weights'] = {
                'traffic': {'name': '交通便利性', 'value': weights.get('traffic', 0.25)},
                'price': {'name': '价格成本', 'value': weights.get('price', 0.25)},
                'area': {'name': '地块规模', 'value': weights.get('area', 0.25)},
                'region': {'name': '区位优势', 'value': weights.get('region', 0.25)}
            }
        except:
            result['weights'] = {}
        
        # 添加区位评分映射
        result['region_scores'] = getattr(self, '_district_score_map', {})
        
        # 为每个地块添加详细评分
        for key, site_entry in result.get('sites', {}).items():
            try:
                site_id = int(site_entry.get('id', 0))
                row = self.site_data.loc[site_id]
                
                # 各项原始分数
                traffic = float(row.get('交通_便利评分(0-10)', 0))
                price = float(row.get('挂牌起始价(万元)', 0))
                area = float(row.get('宗地面积(平方米)', 0))
                addr = str(row.get('宗地坐落', ''))
                region = self._region_score(addr)
                
                # 归一化分数（用于加权计算）
                price_score = self._price_score(row.get('价格_万元/㎡'))
                area_score = self._area_score(row)
                
                site_entry['score_details'] = {
                    'traffic': {'raw': traffic, 'normalized': traffic, 'desc': f'{traffic:.2f}分'},
                    'price': {'raw': price, 'normalized': price_score, 'desc': f'{price_score:.2f}分'},
                    'area': {'raw': area, 'normalized': area_score, 'desc': f'{area_score:.2f}分'},
                    'region': {'raw': region, 'normalized': region, 'desc': f'{region:.2f}分'}
                }
            except:
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
1. **每个地块必须包含**：
   - advantages: 至少3条优势（数组格式）
   - risks: 至少2条风险（数组格式，不能为空）
   - reason: 推荐理由（字符串）
   - score: 评分1-10分

2. **风险分析要点**（必须考虑）：
   - 交通便利性不足
   - 价格成本较高
   - 区域位置不符合需求
   - 用地性质限制
   - 周边配套不完善
   - 开发难度和时间成本

3. **数据引用**：尽量引用具体数据（如距离、评分、价格等）

**重要**：risks字段不能为空，必须至少包含2条风险分析！

请按JSON格式输出。
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
        
        # TSP访问顺序已移除，直接使用评分排序
        ordered_sites = sites
        clusters_order = list(range(len(clusters))) if clusters else []
        
        print("Step 3: 生成推荐报告...")
        recommendation = self.generate_recommendation(ordered_sites, clusters)
        
        print("\n" + "=" * 60)
        print("推荐结果：")
        print("=" * 60)
        print(json.dumps(recommendation, ensure_ascii=False, indent=2))
        
        return recommendation