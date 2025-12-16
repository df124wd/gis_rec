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
