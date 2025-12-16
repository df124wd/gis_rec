"""
多目标优化模块（NSGA-II 完整版）
基于遗传算法的地块组合优化

核心思路：
- 一个解 = 一个方案 = 10个地块的组合
- 优化目标：方案内地块的聚合指标（均值/总和）
- 输出：帕累托前沿（多个最优权衡方案）
- 最终：根据用户偏好选择一个方案，再按总分排序取Top-5
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random
import re


class NSGA2Optimizer:
    """
    NSGA-II 多目标优化器
    
    用于从候选地块中选择最优的地块组合
    """
    
    def __init__(self, 
                 candidate_data: pd.DataFrame,
                 objectives: List[Dict],
                 n_select: int = 10,
                 pop_size: int = 100,
                 n_generations: int = 50):
        """
        Args:
            candidate_data: 候选地块数据（已过滤）
            objectives: 优化目标列表
            n_select: 每个方案选择的地块数量
            pop_size: 种群大小
            n_generations: 进化代数
        """
        self.data = candidate_data.reset_index(drop=True)
        self.n_candidates = len(self.data)
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.n_select = min(n_select, self.n_candidates)
        self.pop_size = pop_size
        self.n_generations = n_generations
        
        # 预计算每个地块的目标值（归一化）
        self._precompute_objectives()
