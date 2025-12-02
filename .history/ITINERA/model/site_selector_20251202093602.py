"""
智能选址推荐系统
基于ITINERA改造 - 精简版
"""

import os
import re
import json
import numpy as np
import concurrent.futures
import pandas as pd

from model.utils.funcs import sample_items
from model.search import SearchEngine
from model.spatial import SpatialHandler


class SiteSelector:
    """
    智能选址推荐系统
    """
    
    # 评价指标体系定义
    METRICS = {
        "traffic": {
            "name": "交通便利性",
            "description": "地铁、公交、停车、火车站等交通设施的便利程度",
            "columns": ["交通_便利评分(0-10)"],
            "weight_default": 0.25
        },
        "price": {
            "name": "价格成本",
            "description": "地块总价和单价的经济性",
            "columns": ["价格_万元/㎡", "挂牌起始价(万元)"],
            "weight_default": 0.20,
            "lower_is_better": True  # 价格越低越好
        },
        "area": {
            "name": "地块规模",
            "description": "地块面积大小是否满足需求",
            "columns": ["宗地面积(平方米)"],
            "weight_default": 0.15
        },
        "location": {
            "name": "区位优势",
            "description": "所在行政区的发展水平和地理位置",
            "columns": ["宗地坐落"],
            "weight_default": 0.15
        },
        "land_use": {
            "name": "用地性质",
            "description": "土地用途是否符合业务需求",
            "columns": ["土地用途"],
            "weight_default": 0.15
        },
        "semantic": {
            "name": "语义匹配度",
            "description": "与用户需求描述的语义相似程度",
            "columns": [],  # 来自向量检索
            "weight_default": 0.10
        }
    }
    
    def __init__(self, user_reqs, min_site_candidate_num=10, 
                 proxy_call=None, city=None, type='zh',
                 dataset_path=None, enable_multi_objective=True,
                 min_distance_meters=0):
        
        self.MODEL = None
        self.min_site_candidate_num = min_site_candidate_num
        self.type = type
        self.proxy = proxy_call
        self.user_reqs = user_reqs
        self.enable_multi_objective = bool(enable_multi_objective)
        self.min_distance_meters = int(min_distance_meters) if min_distance_meters else 0
        self.maxSiteNum = 10
        
        # 解析用户需求
        parsed_request = self.parse_user_request(user_reqs)
        self.parse_site_requirements(parsed_request)
        
        # 用LLM智能推导权重
        self.weights = self.derive_weights_with_llm(user_reqs)
        print(f"[LLM权重推导] {self.weights}")
        
        # 加载数据
        self.load_site_data(city_name=city, dataset_path=dataset_path)
        
        # 初始化检索引擎
        self.search_engine = SearchEngine(
            embedding=self.embedding,
            emb_path=getattr(self, 'emb_path', ''),
            file_path=getattr(self, 'data_path', ''),
            proxy=self.proxy
        )
        
        # 文本分数缓存
        self.text_score_map = {}
        self.text_score_min = 0.0
        self.text_score_max = 1.0

    def derive_weights_with_llm(self, user_reqs: str) -> dict:
        """用LLM智能推导评价指标权重"""
        
        prompt = f"""你是专业的选址顾问。请根据用户需求，推导各评价指标的权重。

## 用户需求
{user_reqs}

## 可用评价指标
1. traffic - 交通便利性（地铁、公交、停车、火车站）
2. price - 价格成本（总价、单价，越低越好）
3. area - 地块规模（面积大小）
4. location - 区位优势（所在行政区的发展水平）
5. land_use - 用地性质（工业/商业/住宅等）
6. semantic - 语义匹配度（与需求描述的相似程度）

## 任务
根据用户需求的侧重点，为每个指标分配权重（0-1之间，总和为1）。

## 输出格式（严格JSON，不要其他内容）
{{
    "weights": {{
        "traffic": 0.25,
        "price": 0.20,
        "area": 0.15,
        "location": 0.15,
        "land_use": 0.15,
        "semantic": 0.10
    }},
    "reasoning": "简要说明权重分配理由"
}}

## 示例分析
- "花都区快递中转站，价格适中，交通便利" → traffic=0.30, price=0.25, land_use=0.20, location=0.10, area=0.10, semantic=0.05
- "天河区20亩工业用地" → location=0.30, land_use=0.25, area=0.20, traffic=0.10, price=0.10, semantic=0.05
- "便宜的大面积仓储用地" → price=0.35, area=0.30, land_use=0.20, traffic=0.10, location=0.03, semantic=0.02
"""
        
        try:
            response = self.proxy.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL
            )
            
            # 尝试解析JSON
            result = json.loads(response)
            weights = result.get('weights', {})
            reasoning = result.get('reasoning', '')
            
            if reasoning:
                print(f"[LLM权重理由] {reasoning}")
            
            # 验证权重完整性
            required_keys = ["traffic", "price", "area", "location", "land_use", "semantic"]
            for key in required_keys:
                if key not in weights:
                    weights[key] = self.METRICS.get(key, {}).get('weight_default', 0.1)
            
            # 归一化
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            print(f"[LLM权重推导失败] {e}，使用默认权重")
            return {
                "traffic": 0.25, "price": 0.20, "area": 0.15,
                "location": 0.15, "land_use": 0.15, "semantic": 0.10
            }

    def parse_user_request(self, user_reqs):
        """解析用户自然语言需求"""
        prompt = self._get_parse_prompt(user_reqs)
        response = self.proxy.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.MODEL
        ).replace("'", '"')
        
        try:
            return json.loads(response)
        except:
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return []

    def _get_parse_prompt(self, user_input):
        """生成需求解析提示词"""
        return f"""请分析用户的选址需求并拆解成结构化格式。

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

请严格按JSON格式返回，不要其他内容。"""

    def parse_site_requirements(self, structured_input):
        """解析结构化需求"""
        self.hard_constraints = []
        self.user_pos_reqs = []
        self.user_neg_reqs = []
        
        for req in structured_input:
            if req.get("mustsee"):
                if req.get("pos"):
                    self.hard_constraints.append({
                        "text": req.get("pos"),
                        "type": req.get("type"),
                        "is_negative": False
                    })
                if req.get("neg"):
                    self.hard_constraints.append({
                        "text": req.get("neg"),
                        "type": req.get("type"),
                        "is_negative": True
                    })
            
            if req.get("pos"):
                self.user_pos_reqs.append(req["pos"])
            if req.get("neg"):
                self.user_neg_reqs.append(req["neg"])
        
        if not self.user_pos_reqs:
            self.user_pos_reqs = [self.user_reqs]
            self.user_neg_reqs = [None]

    def load_site_data(self, city_name, dataset_path=None):
        """加载地块数据"""
        # 解析数据路径
        if dataset_path:
            data_path = dataset_path if os.path.isabs(dataset_path) else os.path.abspath(dataset_path)
            base, ext = os.path.splitext(data_path)
            emb_path = base + ".npy"
        else:
            data_path = os.path.join("model", "data", f'{city_name}_{self.type}.csv')
            emb_path = os.path.join("model", "data", f'{city_name}_{self.type}.npy')
        
        self.data_path = data_path
        self.emb_path = emb_path

        # 读取CSV数据
        self.site_data = pd.read_csv(data_path)
        
        # 标准化列名
        col_mapping = {'经度': 'lon', '纬度': 'lat'}
        self.site_data = self.site_data.rename(columns={k: v for k, v in col_mapping.items() if k in self.site_data.columns})
        
        # 生成标准列
        if 'name' not in self.site_data.columns:
            self.site_data['name'] = self.site_data.get('宗地坐落', self.site_data.index.astype(str))
        if 'id' not in self.site_data.columns:
            self.site_data['id'] = self.site_data.index.astype(int)
        
        # 生成context（用于语义检索）
        if 'context' not in self.site_data.columns:
            parts = []
            if '宗地坐落' in self.site_data.columns:
                parts.append(self.site_data['宗地坐落'].astype(str))
            if '土地用途' in self.site_data.columns:
                parts.append("用途:" + self.site_data['土地用途'].astype(str))
            if '宗地面积(平方米)' in self.site_data.columns:
                parts.append("面积:" + self.site_data['宗地面积(平方米)'].astype(str) + "㎡")
            self.site_data['context'] = "，".join([p for p in parts]) if parts else self.site_data['name']

        # 读取/生成embedding
        if os.path.exists(emb_path):
            self.embedding = np.load(emb_path)
        else:
            se_tmp = SearchEngine(embedding=None, emb_path=emb_path, file_path=data_path, proxy=self.proxy)
            self.embedding = se_tmp.embedding
        
        # 预计算归一化参数
        self._precompute_normalization()
        
        self.site_data = self.site_data.reset_index(drop=True)

    def _precompute_normalization(self):
        """预计算各指标的归一化参数"""
        self._norm_params = {}
        
        # 价格归一化
        if '价格_万元/㎡' in self.site_data.columns:
            s = pd.to_numeric(self.site_data['价格_万元/㎡'], errors='coerce')
            self._norm_params['price'] = {
                'min': float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else 0.0,
                'max': float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else 1.0
            }
        
        # 面积归一化
        if '宗地面积(平方米)' in self.site_data.columns:
            s = pd.to_numeric(self.site_data['宗地面积(平方米)'], errors='coerce')
            self._norm_params['area'] = {
                'min': float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else 0.0,
                'max': float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else 1.0
            }

    def compute_site_score(self, site_id: int) -> dict:
        """计算单个地块的综合评分，返回各维度分数"""
        try:
            row = self.site_data.loc[site_id]
        except:
            return {"total": 5.0, "details": {}}
        
        scores = {}
        
        # 1. 交通便利性 (直接使用数据集中的评分)
        try:
            scores['traffic'] = float(np.clip(row.get('交通_便利评分(0-10)', 5.0), 0, 10))
        except:
            scores['traffic'] = 5.0
        
        # 2. 价格成本 (越低越好，反向归一化)
        try:
            price = float(row.get('价格_万元/㎡', 0))
            params = self._norm_params.get('price', {'min': 0, 'max': 1})
            if params['max'] > params['min']:
                # 价格越低分数越高
                scores['price'] = 10.0 * (1 - (price - params['min']) / (params['max'] - params['min']))
            else:
                scores['price'] = 5.0
            scores['price'] = float(np.clip(scores['price'], 0, 10))
        except:
            scores['price'] = 5.0
        
        # 3. 地块规模 (越大越好)
        try:
            area = float(row.get('宗地面积(平方米)', 0))
            params = self._norm_params.get('area', {'min': 0, 'max': 1})
            if params['max'] > params['min']:
                scores['area'] = 10.0 * (area - params['min']) / (params['max'] - params['min'])
            else:
                scores['area'] = 5.0
            scores['area'] = float(np.clip(scores['area'], 0, 10))
        except:
            scores['area'] = 5.0
        
        # 4. 区位优势 (基于行政区)
        scores['location'] = self._compute_location_score(row)
        
        # 5. 用地性质匹配度
        scores['land_use'] = self._compute_land_use_score(row)
        
        # 6. 语义匹配度 (来自向量检索)
        text_score = self.text_score_map.get(int(site_id), 0)
        if self.text_score_max > self.text_score_min:
            scores['semantic'] = 10.0 * (text_score - self.text_score_min) / (self.text_score_max - self.text_score_min)
        else:
            scores['semantic'] = 5.0
        scores['semantic'] = float(np.clip(scores['semantic'], 0, 10))
        
        # 加权计算总分
        total = sum(scores.get(k, 5.0) * self.weights.get(k, 0.1) for k in self.weights.keys())
        total = float(np.clip(total, 1, 10))
        
        return {"total": total, "details": scores}

    def _compute_location_score(self, row) -> float:
        """计算区位优势分数"""
        # 广州行政区发展水平评分
        district_scores = {
            "天河区": 9.5, "越秀区": 9.3, "海珠区": 9.0, "荔湾区": 8.5,
            "黄埔区": 8.0, "白云区": 7.5, "番禺区": 7.2, "花都区": 6.8,
            "南沙区": 7.0, "增城区": 6.5, "从化区": 6.0,
        }
        
        address = str(row.get('宗地坐落', '') or row.get('name', ''))
        for district, score in district_scores.items():
            if district in address:
                return score
        return 7.0

    def _compute_land_use_score(self, row) -> float:
        """计算用地性质匹配度"""
        land_use = str(row.get('土地用途', ''))
        
        # 检查是否匹配用户需求中的用地类型
        for constraint in self.hard_constraints:
            if constraint.get('type') == '用地类型':
                text = constraint.get('text', '')
                is_negative = constraint.get('is_negative', False)
                
                if is_negative:
                    # 负向约束：包含则扣分
                    if text in land_use:
                        return 2.0
                else:
                    # 正向约束：包含则加分
                    if text in land_use or any(k in land_use for k in ['工业', '商业', '住宅'] if k in text):
                        return 9.0
        
        return 6.0  # 默认中等分数

    def get_candidate_sites(self):
        """检索候选地块"""
        print(f"[检索] 正向需求: {self.user_pos_reqs}")
        print(f"[检索] 负向需求: {self.user_neg_reqs}")
        
        def process_request(pos_req, neg_req):
            top_k = min(self.site_data.shape[0], self.min_site_candidate_num)
            req_sites = self.search_engine.query(
                desc=(pos_req, neg_req if neg_req else ""),
                top_k=top_k
            )
            return req_sites
        
        all_reqs_topk = []
        
        # 并发处理多个需求
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, pos_req in enumerate(self.user_pos_reqs):
                neg_req = self.user_neg_reqs[i] if i < len(self.user_neg_reqs) else None
                futures.append(executor.submit(process_request, pos_req, neg_req))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    req_sites = future.result()
                    if req_sites is not None and len(req_sites) > 0:
                        all_reqs_topk.append(req_sites)
                except Exception as e:
                    print(f"[检索错误] {e}")
        
        if not all_reqs_topk:
            print("[警告] 没有找到任何候选地块")
            return np.empty((0, 2))
        
        # 合并结果
        all_reqs_topk = np.concatenate(all_reqs_topk, axis=0)
        unique_values = np.unique(all_reqs_topk[:, 0])
        result = np.array([
            [value, all_reqs_topk[all_reqs_topk[:, 0] == value][:, 1].sum()]
            for value in unique_values
        ])
        
        # 缓存文本分数
        self.text_score_map = {int(v): float(s) for v, s in result.tolist()}
        self.text_score_min = float(result[:, 1].min()) if result.size else 0.0
        self.text_score_max = float(result[:, 1].max()) if result.size else 1.0
        
        print(f"[检索] 找到 {len(result)} 个候选地块")
        
        return result[result[:, 1].argsort()[::-1]]

    def rank_candidates(self, candidates: np.ndarray) -> list:
        """对候选地块进行综合评分排序"""
        scored_sites = []
        
        for site_id, _ in candidates:
            site_id = int(site_id)
            score_result = self.compute_site_score(site_id)
            scored_sites.append({
                'id': site_id,
                'total_score': score_result['total'],
                'details': score_result['details']
            })
        
        # 按总分排序
        scored_sites.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 打印评分详情
        print("\n[评分排序] Top-10 地块评分详情:")
        print("-" * 100)
        header = f"{'排名':<4} {'ID':<4} {'总分':<6} {'交通':<6} {'价格':<6} {'面积':<6} {'区位':<6} {'用地':<6} {'语义':<6}"
        print(header)
        print("-" * 100)
        
        for i, site in enumerate(scored_sites[:10]):
            d = site['details']
            print(f"{i+1:<4} {site['id']:<4} {site['total_score']:<6.2f} "
                  f"{d.get('traffic', 0):<6.2f} {d.get('price', 0):<6.2f} "
                  f"{d.get('area', 0):<6.2f} {d.get('location', 0):<6.2f} "
                  f"{d.get('land_use', 0):<6.2f} {d.get('semantic', 0):<6.2f}")
        
        return scored_sites

    def apply_spatial_diversity(self, ranked_sites: list) -> list:
        """应用空间多样性过滤（最小间距）"""
        if self.min_distance_meters <= 0:
            return ranked_sites[:self.maxSiteNum]
        
        def haversine(lon1, lat1, lon2, lat2):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371000.0
            dlon, dlat = radians(lon2 - lon1), radians(lat2 - lat1)
            a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
            return R * 2 * atan2(sqrt(a), sqrt(1-a))
        
        selected = []
        for site in ranked_sites:
            sid = site['id']
            try:
                row = self.site_data.loc[sid]
                lon, lat = float(row['lon']), float(row['lat'])
                
                too_close = False
                for s in selected:
                    r2 = self.site_data.loc[s['id']]
                    if haversine(lon, lat, float(r2['lon']), float(r2['lat'])) < self.min_distance_meters:
                        too_close = True
                        break
                
                if not too_close:
                    selected.append(site)
            except:
                selected.append(site)
            
            if len(selected) >= self.maxSiteNum:
                break
        
        return selected

    def generate_recommendation(self, ranked_sites: list):
        """生成推荐报告"""
        
        # 准备候选地块信息
        context_string = ""
        for i, site in enumerate(ranked_sites[:self.maxSiteNum]):
            site_id = site['id']
            row = self.site_data.loc[site_id]
            context = row.get('context', row.get('name', f'地块{site_id}'))
            context_string += f'序号{i+1}: "{context[:150]}"\n'
        
        prompt = f"""你是专业的选址顾问。请根据候选地块和用户需求，生成推荐报告。

### 用户需求
{self.user_reqs}

### 候选地块
{context_string}

### 输出格式（严格JSON）
{{
    "summary": "总体推荐理由（50字以内）",
    "sites": {{
        "1": {{
            "name": "地块名称",
            "reason": "推荐理由（30字以内）",
            "advantages": ["优势1", "优势2", "优势3"],
            "risks": ["风险1", "风险2"]
        }}
    }}
}}

要求：
1. 每个地块必须有3条优势和2条风险
2. 优势和风险要具体，引用数据
3. 严格按JSON格式输出"""
        
        response = self.proxy.chat(
            messages=[
                {"role": "system", "content": "你是专业的选址顾问"},
                {"role": "user", "content": prompt}
            ],
            model=self.MODEL
        )
        
        try:
            result = json.loads(response)
        except:
            result = {"summary": "推荐完成", "sites": {}}
        
        # 补充坐标和分数
        features = []
        enriched_sites = {}
        
        for i, site in enumerate(ranked_sites[:self.maxSiteNum]):
            site_id = site['id']
            row = self.site_data.loc[site_id]
            key = str(i + 1)
            
            site_entry = result.get('sites', {}).get(key, {})
            site_entry['id'] = str(site_id)
            site_entry['lat'] = float(row['lat'])
            site_entry['lon'] = float(row['lon'])
            site_entry['score'] = round(site['total_score'], 2)
            site_entry['score_details'] = site['details']
            
            # 使用数据集中的名称
            site_entry['name'] = str(row.get('name', row.get('宗地坐落', f'地块{key}')))
            
            # 确保有优势和风险
            if not site_entry.get('advantages'):
                site_entry['advantages'] = [row.get('context', '')[:100]]
            if not site_entry.get('risks'):
                site_entry['risks'] = ["建议实地考察确认"]
            
            enriched_sites[key] = site_entry
            
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [site_entry['lon'], site_entry['lat']]},
                "properties": {"index": i + 1, "id": str(site_id), "name": site_entry['name'], "score": site_entry['score']}
            })
        
        result['sites'] = enriched_sites
        result['features'] = features
        result['geojson'] = {"type": "FeatureCollection", "features": features}
        
        if features:
            lons = [f['geometry']['coordinates'][0] for f in features]
            lats = [f['geometry']['coordinates'][1] for f in features]
            result['center'] = {"lon": sum(lons)/len(lons), "lat": sum(lats)/len(lats)}
        
        return result

    def solve(self):
        """执行完整的选址推荐流程"""
        
        print("\n" + "="*60)
        print("Step 1: 检索候选地块...")
        print("="*60)
        candidates = self.get_candidate_sites()
        
        if candidates.size == 0:
            return {"error": "未找到符合条件的地块"}
        
        print("\n" + "="*60)
        print("Step 2: 综合评分排序...")
        print("="*60)
        ranked_sites = self.rank_candidates(candidates)
        
        print("\n" + "="*60)
        print("Step 3: 空间多样性过滤...")
        print("="*60)
        final_sites = self.apply_spatial_diversity(ranked_sites)
        print(f"[结果] 最终选择 {len(final_sites)} 个地块")
        
        print("\n" + "="*60)
        print("Step 4: 生成推荐报告...")
        print("="*60)
        recommendation = self.generate_recommendation(final_sites)
        
        print("\n" + "="*60)
        print("推荐完成！")
        print("="*60)
        
        return recommendation
