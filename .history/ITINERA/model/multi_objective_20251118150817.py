"""
多目标优化模块
基于帕累托前沿的地块推荐
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class MultiObjectiveOptimizer:
    """
    多目标优化器
    使用帕累托支配关系筛选非支配解集
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: 候选地块数据（DataFrame）
        """
        self.data = data
    
    def pareto_front(self, 
                     objectives: List[Dict],
                     candidates: List[int] = None) -> List[int]:
        """
        计算帕累托前沿（非支配解集）
        
        Args:
            objectives: 优化目标列表，例如：
                [
                    {'name': '交通_便利评分(0-10)', 'maximize': True},
                    {'name': '价格_万元/㎡', 'maximize': False},  # 价格越低越好
                    {'name': '宗地面积(平方米)', 'maximize': True}
                ]
            candidates: 候选地块索引列表，None表示全部
        
        Returns:
            pareto_front: 帕累托前沿地块索引列表
        """
        if candidates is None:
            candidates = self.data.index.tolist()
        
        # 提取目标值矩阵
        obj_matrix = []
        for idx in candidates:
            row = self.data.loc[idx]
            values = []
            for obj in objectives:
                val = float(row[obj['name']])
                # 处理inf值
                if np.isinf(val):
                    val = 1e9 if val > 0 else -1e9
                # 如果是最小化目标，取负值
                if not obj['maximize']:
                    val = -val
                values.append(val)
            obj_matrix.append(values)
        
        obj_matrix = np.array(obj_matrix)
        
        # 计算帕累托前沿
        pareto_indices = []
        for i, c1 in enumerate(candidates):
            dominated = False
            for j, c2 in enumerate(candidates):
                if i != j and self._dominates(obj_matrix[j], obj_matrix[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(c1)
        
        return pareto_indices
    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        判断a是否支配b（所有维度不差，至少一个维度更好）
        
        Args:
            a, b: 目标值向量
        
        Returns:
            True if a dominates b
        """
        better_in_any = False
        for i in range(len(a)):
            if a[i] < b[i]:
                return False  # a在某个维度更差
            if a[i] > b[i]:
                better_in_any = True  # a在某个维度更好
        return better_in_any
    
    def rank_pareto_front(self,
                          pareto_indices: List[int],
                          objectives: List[Dict],
                          weights: Dict[str, float] = None) -> List[Tuple[int, float]]:
        """
        对帕累托前沿进行排序（用于展示）
        
        Args:
            pareto_indices: 帕累托前沿地块索引
            objectives: 优化目标列表
            weights: 权重字典（可选），用于加权排序
        
        Returns:
            ranked: [(index, score), ...] 排序后的地块列表
        """
        if weights is None:
            # 默认等权重
            weights = {obj['name']: 1.0 / len(objectives) for obj in objectives}
        
        scored = []
        for idx in pareto_indices:
            row = self.data.loc[idx]
            score = 0.0
            for obj in objectives:
                val = float(row[obj['name']])
                if np.isinf(val):
                    val = 0.0  # inf值记为0分
                else:
                    # 归一化到[0, 1]
                    col_values = self.data[obj['name']].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(col_values) > 0:
                        vmin, vmax = col_values.min(), col_values.max()
                        if vmax > vmin:
                            val = (val - vmin) / (vmax - vmin)
                        else:
                            val = 0.5
                    else:
                        val = 0.5
                    
                    # 如果是最小化目标，取反
                    if not obj['maximize']:
                        val = 1.0 - val
                
                score += weights.get(obj['name'], 1.0) * val
            
            scored.append((idx, score))
        
        # 按分数降序排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def explain_pareto(self,
                       pareto_indices: List[int],
                       objectives: List[Dict]) -> Dict:
        """
        生成帕累托前沿的解释
        
        Returns:
            explanation: {
                'count': 前沿地块数量,
                'diversity': 多样性指标,
                'trade_offs': 权衡分析
            }
        """
        if not pareto_indices:
            return {'count': 0, 'diversity': 0.0, 'trade_offs': []}
        
        # 提取目标值
        obj_values = []
        for idx in pareto_indices:
            row = self.data.loc[idx]
            values = {}
            for obj in objectives:
                val = float(row[obj['name']])
                if not np.isinf(val):
                    values[obj['name']] = val
            obj_values.append(values)
        
        # 计算多样性（目标值的标准差）
        diversity_scores = []
        for obj in objectives:
            vals = [v.get(obj['name'], 0) for v in obj_values]
            if len(vals) > 1:
                diversity_scores.append(np.std(vals))
        diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
        
        # 权衡分析（找出极端点）
        trade_offs = []
        for obj in objectives:
            vals = [(i, v.get(obj['name'], 0)) for i, v in enumerate(obj_values)]
            if obj['maximize']:
                best_idx, best_val = max(vals, key=lambda x: x[1])
                trade_offs.append({
                    'objective': obj['name'],
                    'best_index': pareto_indices[best_idx],
                    'best_value': best_val,
                    'type': '最大化'
                })
            else:
                best_idx, best_val = min(vals, key=lambda x: x[1])
                trade_offs.append({
                    'objective': obj['name'],
                    'best_index': pareto_indices[best_idx],
                    'best_value': best_val,
                    'type': '最小化'
                })
        
        return {
            'count': len(pareto_indices),
            'diversity': diversity,
            'trade_offs': trade_offs
        }


# 使用示例
if __name__ == '__main__':
    # 加载数据
    df = pd.read_csv('../model/data/land_transactions_with_coordinates_metrics.csv')
    
    # 创建优化器
    optimizer = MultiObjectiveOptimizer(df)
    
    # 定义优化目标
    objectives = [
        {'name': '交通_便利评分(0-10)', 'maximize': True},
        {'name': '价格_万元/㎡', 'maximize': False},  # 价格越低越好
        {'name': '宗地面积(平方米)', 'maximize': True}
    ]
    
    # 计算帕累托前沿
    pareto = optimizer.pareto_front(objectives)
    print(f"帕累托前沿包含 {len(pareto)} 个地块（共{len(df)}个候选）")
    
    # 排序展示
    ranked = optimizer.rank_pareto_front(pareto, objectives)
    print("\n排序后的帕累托前沿：")
    for idx, score in ranked[:5]:
        row = df.loc[idx]
        print(f"地块{idx}: 交通{row['交通_便利评分(0-10)']:.1f}分, "
              f"价格{row['价格_万元/㎡']:.2f}万/㎡, "
              f"面积{row['宗地面积(平方米)']:.0f}㎡, "
              f"综合分{score:.2f}")
    
    # 解释
    explanation = optimizer.explain_pareto(pareto, objectives)
    print(f"\n多样性指标: {explanation['diversity']:.2f}")
    print("\n权衡分析:")
    for trade_off in explanation['trade_offs']:
        print(f"- {trade_off['objective']}最优: 地块{trade_off['best_index']} "
              f"({trade_off['best_value']:.2f})")
