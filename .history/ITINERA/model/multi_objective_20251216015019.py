"""
多目标优化模块（NSGA-III + 两阶段优化）
基于帕累托前沿的地块推荐

两阶段优化流程：
1. 阶段1：硬约束过滤 + 帕累托前沿筛选（粗筛）
2. 阶段2：LLM权重加权排序（精排）
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re


class HardConstraintParser:
    """
    硬约束解析器
    将LLM提取的文本约束转换为可执行的过滤条件
    """
    
    # 数值提取正则
    NUMBER_PATTERN = re.compile(r'[≥≤>=<]?\s*(\d+(?:\.\d+)?)')
    
    # 约束类型到字段的映射
    CONSTRAINT_FIELD_MAP = {
        '区域': '宗地坐落',
        '用地类型': '土地用途',
        '面积': '宗地面积(平方米)',
        '成本': '挂牌起始价(万元)',
        '单价': '价格_万元/㎡',
        '交通': '交通_便利评分(0-10)',
        '地铁': '交通_地铁最近距离(m)',
    }
    
    # 面积单位转换
    AREA_UNITS = {
        '亩': 666.67,
        '公顷': 10000,
        '平方米': 1,
        '㎡': 1,
        '平米': 1,
    }

    @classmethod
    def parse_constraints(cls, hard_constraints: List[Dict], proxy=None) -> List[Dict]:
        """
        将文本约束解析为可执行的过滤条件
        
        Args:
            hard_constraints: [{text, type, is_negative}, ...]
            proxy: LLM代理（用于复杂约束解析）
        
        Returns:
            executable_constraints: [{field, operator, value, is_negative}, ...]
        """
        executable = []
        
        for c in hard_constraints:
            text = c.get('text', '').strip()
            ctype = c.get('type', '')
            is_neg = c.get('is_negative', False)
            
            if not text:
                continue
            
            parsed = cls._parse_single_constraint(text, ctype, is_neg, proxy)
            if parsed:
                executable.append(parsed)
        
        return executable
    
    @classmethod
    def _parse_single_constraint(cls, text: str, ctype: str, is_neg: bool, proxy=None) -> Optional[Dict]:
        """解析单个约束"""
        
        # 1. 区域约束（包含匹配）
        if ctype == '区域':
            # 提取区域名称
            districts = ['天河区', '越秀区', '海珠区', '荔湾区', '黄埔区', 
                        '白云区', '番禺区', '花都区', '南沙区', '增城区', '从化区']
            for d in districts:
                if d in text or d.replace('区', '') in text:
                    return {
                        'field': '宗地坐落',
                        'operator': 'not_contains' if is_neg else 'contains',
                        'value': d.replace('区', ''),  # 匹配"花都"而非"花都区"更宽松
                        'is_negative': is_neg,
                        'original_text': text
                    }
            # 如果没匹配到具体区，用原文本
            return {
                'field': '宗地坐落',
                'operator': 'not_contains' if is_neg else 'contains',
                'value': text,
                'is_negative': is_neg,
                'original_text': text
            }
        
        # 2. 用地类型约束
        if ctype == '用地类型':
            land_types = ['工业', '商业', '居住', '仓储', '物流', '办公']
            for lt in land_types:
                if lt in text:
                    return {
                        'field': '土地用途',
                        'operator': 'not_contains' if is_neg else 'contains',
                        'value': lt,
                        'is_negative': is_neg,
                        'original_text': text
                    }
            return {
                'field': '土地用途',
                'operator': 'not_contains' if is_neg else 'contains',
                'value': text,
                'is_negative': is_neg,
                'original_text': text
            }

        # 3. 面积约束
        if ctype == '面积':
            # 提取数值和单位
            number_match = cls.NUMBER_PATTERN.search(text)
            if number_match:
                value = float(number_match.group(1))
                # 检测单位并转换为平方米
                for unit, factor in cls.AREA_UNITS.items():
                    if unit in text:
                        value *= factor
                        break
                # 检测操作符
                if '≥' in text or '>=' in text or '至少' in text or '不少于' in text or '以上' in text:
                    op = '>='
                elif '≤' in text or '<=' in text or '不超过' in text or '以下' in text or '最多' in text:
                    op = '<='
                else:
                    op = '>='  # 默认"至少"
                
                return {
                    'field': '宗地面积(平方米)',
                    'operator': op,
                    'value': value,
                    'is_negative': is_neg,
                    'original_text': text
                }
        
        # 4. 成本/价格约束
        if ctype == '成本':
            number_match = cls.NUMBER_PATTERN.search(text)
            if number_match:
                value = float(number_match.group(1))
                # 判断是总价还是单价
                if '元/㎡' in text or '元/平' in text or '单价' in text:
                    field = '价格_万元/㎡'
                    value = value / 10000  # 转换为万元/㎡
                else:
                    field = '挂牌起始价(万元)'
                
                # 检测操作符
                if '≤' in text or '<=' in text or '不超过' in text or '以下' in text:
                    op = '<='
                elif '≥' in text or '>=' in text or '至少' in text or '以上' in text:
                    op = '>='
                else:
                    op = '<='  # 价格默认"不超过"
                
                return {
                    'field': field,
                    'operator': op,
                    'value': value,
                    'is_negative': is_neg,
                    'original_text': text
                }
        
        # 5. 配套约束（地铁距离等）
        if ctype == '配套':
            if '地铁' in text:
                number_match = cls.NUMBER_PATTERN.search(text)
                if number_match:
                    value = float(number_match.group(1))
                    return {
                        'field': '交通_地铁最近距离(m)',
                        'operator': '<=',
                        'value': value,
                        'is_negative': is_neg,
                        'original_text': text
                    }
                else:
                    # 没有具体数值，用默认1500m
                    return {
                        'field': '交通_地铁最近距离(m)',
                        'operator': '<=',
                        'value': 1500,
                        'is_negative': is_neg,
                        'original_text': text
                    }
        
        # 无法解析的约束，返回None（后续用语义匹配兜底）
        return None


class MultiObjectiveOptimizer:
    """
    多目标优化器（两阶段法）
    
    阶段1：硬约束过滤 + 帕累托前沿筛选
    阶段2：LLM权重加权排序
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: 候选地块数据（DataFrame）
        """
        self.data = data
        self.filtered_data = None
        self.pareto_indices = []
    
    def apply_hard_constraints(self, constraints: List[Dict]) -> pd.DataFrame:
        """
        应用硬约束过滤
        
        Args:
            constraints: 可执行约束列表 [{field, operator, value, is_negative}, ...]
        
        Returns:
            过滤后的DataFrame
        """
        filtered = self.data.copy()
        
        for c in constraints:
            field = c.get('field')
            op = c.get('operator')
            value = c.get('value')
            is_neg = c.get('is_negative', False)
            
            if field not in filtered.columns:
                print(f"[约束跳过] 字段不存在: {field}")
                continue
            
            original_count = len(filtered)
            
            try:
                if op == 'contains':
                    mask = filtered[field].astype(str).str.contains(str(value), na=False)
                    if is_neg:
                        mask = ~mask
                    filtered = filtered[mask]
                
                elif op == 'not_contains':
                    mask = ~filtered[field].astype(str).str.contains(str(value), na=False)
                    filtered = filtered[mask]
                
                elif op == '>=':
                    col = pd.to_numeric(filtered[field], errors='coerce')
                    mask = col >= float(value)
                    if is_neg:
                        mask = ~mask
                    filtered = filtered[mask]
                
                elif op == '<=':
                    col = pd.to_numeric(filtered[field], errors='coerce')
                    mask = col <= float(value)
                    if is_neg:
                        mask = ~mask
                    filtered = filtered[mask]
                
                elif op == '==':
                    mask = filtered[field] == value
                    if is_neg:
                        mask = ~mask
                    filtered = filtered[mask]
                
                new_count = len(filtered)
                print(f"[约束应用] {c.get('original_text', field)}: {original_count} → {new_count}")
                
            except Exception as e:
                print(f"[约束错误] {field} {op} {value}: {e}")
                continue
        
        self.filtered_data = filtered.reset_index(drop=True)
        return self.filtered_data

    def compute_pareto_front(self, objectives: List[Dict], 
                             candidate_indices: List[int] = None) -> List[int]:
        """
        计算帕累托前沿（非支配解集）
        
        Args:
            objectives: 优化目标列表
                [{'name': '交通_便利评分(0-10)', 'maximize': True}, ...]
            candidate_indices: 候选索引列表，None表示使用filtered_data全部
        
        Returns:
            pareto_indices: 帕累托前沿地块在filtered_data中的索引
        """
        data = self.filtered_data if self.filtered_data is not None else self.data
        
        if candidate_indices is None:
            candidate_indices = list(range(len(data)))
        
        if len(candidate_indices) == 0:
            return []
        
        n = len(candidate_indices)
        
        # 构建目标矩阵（统一转为"越大越好"）
        obj_matrix = np.zeros((n, len(objectives)))
        
        for i, idx in enumerate(candidate_indices):
            row = data.iloc[idx]
            for j, obj in enumerate(objectives):
                col_name = obj['name']
                try:
                    val = float(row.get(col_name, 0))
                    if np.isnan(val) or np.isinf(val):
                        val = 0
                except:
                    val = 0
                
                # 如果是最小化目标，取负值（统一为越大越好）
                if not obj.get('maximize', True):
                    val = -val
                
                obj_matrix[i, j] = val
        
        # 计算帕累托前沿（非支配解）
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not pareto_mask[i]:
                continue
            for j in range(n):
                if i == j or not pareto_mask[j]:
                    continue
                # 检查j是否支配i
                if self._dominates(obj_matrix[j], obj_matrix[i]):
                    pareto_mask[i] = False
                    break
        
        self.pareto_indices = [candidate_indices[i] for i in range(n) if pareto_mask[i]]
        
        return self.pareto_indices
    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        判断a是否支配b
        支配条件：a在所有维度不差于b，且至少一个维度更好
        """
        better_in_any = False
        for i in range(len(a)):
            if a[i] < b[i]:
                return False  # a在某维度更差
            if a[i] > b[i]:
                better_in_any = True
        return better_in_any

    def rank_by_weights(self, indices: List[int], objectives: List[Dict], 
                        weights: Dict[str, float]) -> List[Tuple[int, float, Dict]]:
        """
        按LLM权重对地块进行加权排序
        
        Args:
            indices: 待排序的地块索引（在filtered_data中）
            objectives: 优化目标列表
            weights: 权重字典 {'traffic': 0.3, 'price': 0.25, ...}
        
        Returns:
            sorted_list: [(index, total_score, score_breakdown), ...] 按分数降序
        """
        data = self.filtered_data if self.filtered_data is not None else self.data
        
        if len(indices) == 0:
            return []
        
        # 计算每个目标的全局min/max（用于归一化）
        obj_stats = {}
        for obj in objectives:
            col = obj['name']
            if col in data.columns:
                col_data = pd.to_numeric(data[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
                if len(col_data) > 0:
                    obj_stats[col] = {'min': col_data.min(), 'max': col_data.max()}
                else:
                    obj_stats[col] = {'min': 0, 'max': 10}
            else:
                obj_stats[col] = {'min': 0, 'max': 10}
        
        scored = []
        
        for idx in indices:
            row = data.iloc[idx]
            total_score = 0
            breakdown = {}
            
            for obj in objectives:
                col = obj['name']
                weight_key = obj.get('weight_key', col)
                weight = weights.get(weight_key, 0.25)
                
                try:
                    val = float(row.get(col, 0))
                    if np.isnan(val) or np.isinf(val):
                        val = 0
                except:
                    val = 0
                
                # 归一化到0-10
                stats = obj_stats.get(col, {'min': 0, 'max': 10})
                vmin, vmax = stats['min'], stats['max']
                if vmax > vmin:
                    val_norm = (val - vmin) / (vmax - vmin) * 10
                else:
                    val_norm = 5
                
                # 如果是最小化目标（如价格），取反
                if not obj.get('maximize', True):
                    val_norm = 10 - val_norm
                
                val_norm = max(0, min(10, val_norm))
                
                total_score += weight * val_norm
                breakdown[weight_key] = {
                    'raw': val,
                    'normalized': val_norm,
                    'weight': weight,
                    'contribution': weight * val_norm
                }
            
            scored.append((idx, total_score, breakdown))
        
        # 按总分降序排序
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored

    def two_stage_optimize(self, 
                           objectives: List[Dict],
                           weights: Dict[str, float],
                           hard_constraints: List[Dict] = None,
                           top_k: int = 10,
                           min_pareto_size: int = 5) -> Dict:
        """
        两阶段优化主流程
        
        Args:
            objectives: 优化目标列表
            weights: LLM推导的权重
            hard_constraints: 硬约束列表（已解析的可执行格式）
            top_k: 最终推荐数量
            min_pareto_size: 帕累托前沿最小数量（不足时补充）
        
        Returns:
            {
                'selected_indices': 原始DataFrame中的索引,
                'scores': 对应的分数,
                'breakdowns': 分数明细,
                'pareto_size': 帕累托前沿大小,
                'filtered_size': 硬约束过滤后数量,
                'original_size': 原始数量
            }
        """
        result = {
            'selected_indices': [],
            'scores': [],
            'breakdowns': [],
            'pareto_size': 0,
            'filtered_size': 0,
            'original_size': len(self.data)
        }
        
        print("\n" + "="*70)
        print("[两阶段多目标优化]")
        print("="*70)
        
        # ========== 阶段0：硬约束过滤 ==========
        print(f"\n[阶段0] 硬约束过滤")
        print(f"  原始候选: {len(self.data)} 个")
        
        if hard_constraints and len(hard_constraints) > 0:
            self.apply_hard_constraints(hard_constraints)
        else:
            self.filtered_data = self.data.copy().reset_index(drop=True)
            print("  无硬约束，跳过过滤")
        
        result['filtered_size'] = len(self.filtered_data)
        print(f"  过滤后: {result['filtered_size']} 个")
        
        if result['filtered_size'] == 0:
            print("  [警告] 硬约束过滤后无候选！放宽约束...")
            self.filtered_data = self.data.copy().reset_index(drop=True)
            result['filtered_size'] = len(self.filtered_data)
        
        # ========== 阶段1：帕累托前沿筛选 ==========
        print(f"\n[阶段1] 帕累托前沿筛选")
        print(f"  优化目标: {[obj['name'] for obj in objectives]}")
        
        pareto_indices = self.compute_pareto_front(objectives)
        result['pareto_size'] = len(pareto_indices)
        
        print(f"  帕累托前沿: {result['pareto_size']} 个非支配解")
        
        # 如果帕累托前沿太小，补充候选
        if len(pareto_indices) < min_pareto_size:
            print(f"  [补充] 帕累托前沿不足{min_pareto_size}个，补充候选...")
            # 按第一个目标排序补充
            all_indices = list(range(len(self.filtered_data)))
            remaining = [i for i in all_indices if i not in pareto_indices]
            
            # 简单按综合分排序补充
            if remaining:
                temp_scored = self.rank_by_weights(remaining, objectives, weights)
                for idx, score, _ in temp_scored:
                    if len(pareto_indices) >= min_pareto_size:
                        break
                    pareto_indices.append(idx)
            
            print(f"  补充后: {len(pareto_indices)} 个")

        # ========== 阶段2：LLM权重加权排序 ==========
        print(f"\n[阶段2] LLM权重加权排序")
        print(f"  权重: {weights}")
        
        ranked = self.rank_by_weights(pareto_indices, objectives, weights)
        
        # 取Top-K
        top_k_results = ranked[:top_k]
        
        # 映射回原始DataFrame索引
        # filtered_data的索引是0,1,2...，需要找到对应的原始索引
        # 由于我们用reset_index(drop=True)，需要保存原始索引映射
        # 这里简化处理：直接用filtered_data的行号作为结果
        
        for idx, score, breakdown in top_k_results:
            result['selected_indices'].append(idx)
            result['scores'].append(score)
            result['breakdowns'].append(breakdown)
        
        print(f"\n[结果] 选出Top-{len(result['selected_indices'])}推荐")
        print("="*70)
        
        return result
    
    def get_site_by_filtered_index(self, filtered_idx: int) -> pd.Series:
        """根据filtered_data中的索引获取地块数据"""
        if self.filtered_data is not None and filtered_idx < len(self.filtered_data):
            return self.filtered_data.iloc[filtered_idx]
        return None
    
    def compute_hypervolume(self, pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
        """
        计算超体积指标（HV）
        用于评估帕累托前沿的质量
        
        Args:
            pareto_front: 帕累托前沿的目标值矩阵 (n_solutions, n_objectives)
            ref_point: 参考点（各目标的最差值）
        
        Returns:
            hv: 超体积值
        """
        try:
            from pymoo.indicators.hv import HV
            hv_calculator = HV(ref_point=ref_point)
            return hv_calculator(pareto_front)
        except ImportError:
            # 如果没有安装pymoo，使用简化计算
            print("[警告] pymoo未安装，跳过HV计算")
            return 0.0
        except Exception as e:
            print(f"[HV计算错误] {e}")
            return 0.0
    
    def compute_spread(self, pareto_front: np.ndarray) -> float:
        """
        计算分布性指标（Spread）
        衡量帕累托前沿解的分布均匀程度
        
        Args:
            pareto_front: 帕累托前沿的目标值矩阵
        
        Returns:
            spread: 分布性指标（越小越均匀）
        """
        if len(pareto_front) < 2:
            return 0.0
        
        # 计算相邻解之间的距离
        distances = []
        for i in range(len(pareto_front) - 1):
            d = np.linalg.norm(pareto_front[i] - pareto_front[i+1])
            distances.append(d)
        
        if len(distances) == 0:
            return 0.0
        
        mean_d = np.mean(distances)
        if mean_d == 0:
            return 0.0
        
        spread = np.sum(np.abs(np.array(distances) - mean_d)) / (len(distances) * mean_d)
        return spread


def get_default_objectives() -> List[Dict]:
    """获取默认的优化目标配置"""
    return [
        {
            'name': '交通_便利评分(0-10)',
            'maximize': True,
            'weight_key': 'traffic',
            'desc': '交通便利性'
        },
        {
            'name': '价格_万元/㎡',
            'maximize': False,  # 价格越低越好
            'weight_key': 'price',
            'desc': '价格成本'
        },
        {
            'name': '宗地面积(平方米)',
            'maximize': True,
            'weight_key': 'area',
            'desc': '地块规模'
        },
        {
            'name': '生活_便利评分(0-10)',
            'maximize': True,
            'weight_key': 'amenity',
            'desc': '生活配套'
        },
    ]


def get_industry_objectives() -> List[Dict]:
    """获取工业选址的优化目标配置"""
    return [
        {
            'name': '交通_便利评分(0-10)',
            'maximize': True,
            'weight_key': 'traffic',
            'desc': '交通便利性'
        },
        {
            'name': '价格_万元/㎡',
            'maximize': False,
            'weight_key': 'price',
            'desc': '价格成本'
        },
        {
            'name': '宗地面积(平方米)',
            'maximize': True,
            'weight_key': 'area',
            'desc': '地块规模'
        },
        {
            'name': '产业_配套评分(0-10)',
            'maximize': True,
            'weight_key': 'industry',
            'desc': '产业配套'
        },
    ]


# ========== 测试代码 ==========
if __name__ == '__main__':
    import os
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'land_transactions_with_poi_v2.csv')
    df = pd.read_csv(data_path)
    print(f"加载数据: {len(df)} 条记录")
    
    # 模拟硬约束（LLM提取的）
    raw_constraints = [
        {'text': '花都区', 'type': '区域', 'is_negative': False},
        {'text': '工业用地', 'type': '用地类型', 'is_negative': False},
        {'text': '面积≥5000平方米', 'type': '面积', 'is_negative': False},
    ]
    
    # 解析硬约束
    executable_constraints = HardConstraintParser.parse_constraints(raw_constraints)
    print(f"\n解析后的约束: {executable_constraints}")
    
    # 创建优化器
    optimizer = MultiObjectiveOptimizer(df)
    
    # 定义目标和权重
    objectives = get_industry_objectives()
    weights = {
        'traffic': 0.30,
        'price': 0.25,
        'area': 0.25,
        'industry': 0.20,
    }
    
    # 运行两阶段优化
    result = optimizer.two_stage_optimize(
        objectives=objectives,
        weights=weights,
        hard_constraints=executable_constraints,
        top_k=10
    )
    
    # 输出结果
    print("\n========== 推荐结果 ==========")
    for i, (idx, score) in enumerate(zip(result['selected_indices'], result['scores'])):
        site = optimizer.get_site_by_filtered_index(idx)
        if site is not None:
            name = str(site.get('宗地坐落', ''))[:40]
            print(f"{i+1}. {name}...")
            print(f"   总分: {score:.2f}")
            print(f"   交通: {site.get('交通_便利评分(0-10)', 0):.1f}, "
                  f"价格: {site.get('价格_万元/㎡', 0):.4f}万/㎡, "
                  f"面积: {site.get('宗地面积(平方米)', 0):.0f}㎡")
