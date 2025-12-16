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

    def _precompute_objectives(self):
        """预计算每个地块在各目标上的归一化值"""
        self.obj_values = np.zeros((self.n_candidates, self.n_obj))
        self.obj_raw = np.zeros((self.n_candidates, self.n_obj))
        
        for j, obj in enumerate(self.objectives):
            col = obj['name']
            if col not in self.data.columns:
                print(f"[警告] 目标列不存在: {col}")
                continue
            
            # 获取原始值
            raw = pd.to_numeric(self.data[col], errors='coerce').fillna(0).values
            self.obj_raw[:, j] = raw
            
            # 归一化到[0,1]
            vmin, vmax = np.nanmin(raw), np.nanmax(raw)
            if vmax > vmin:
                normalized = (raw - vmin) / (vmax - vmin)
            else:
                normalized = np.full_like(raw, 0.5)
            
            # 如果是最小化目标，取反（统一为越大越好）
            if not obj.get('maximize', True):
                normalized = 1 - normalized
            
            self.obj_values[:, j] = normalized
    
    def _init_population(self) -> np.ndarray:
        """初始化种群：随机生成pop_size个方案"""
        population = []
        for _ in range(self.pop_size):
            # 随机选择n_select个不重复的地块索引
            individual = np.random.choice(self.n_candidates, self.n_select, replace=False)
            individual = np.sort(individual)  # 排序便于比较
            population.append(individual)
        return np.array(population)
    
    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        """
        评估种群中每个方案的目标值
        
        Args:
            population: (pop_size, n_select) 每行是一个方案（地块索引列表）
        
        Returns:
            fitness: (pop_size, n_obj) 每行是一个方案的目标值向量
        """
        fitness = np.zeros((len(population), self.n_obj))
        
        for i, individual in enumerate(population):
            # 获取该方案选中的地块的目标值
            selected_values = self.obj_values[individual]  # (n_select, n_obj)
            
            # 聚合：使用均值
            fitness[i] = np.mean(selected_values, axis=0)
        
        return fitness
