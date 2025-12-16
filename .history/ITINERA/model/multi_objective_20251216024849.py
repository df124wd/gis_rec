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

    def _fast_non_dominated_sort(self, fitness: np.ndarray) -> List[List[int]]:
        """
        快速非支配排序
        
        Returns:
            fronts: 分层的前沿列表，fronts[0]是帕累托前沿
        """
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)  # 被支配次数
        dominated_set = [[] for _ in range(n)]  # 支配的解集合
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fitness[i], fitness[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(fitness[j], fitness[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        # 找出第一前沿（不被任何解支配的）
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # 逐层构建后续前沿
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # 去掉最后一个空列表
    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """判断a是否支配b（所有维度不差，至少一个更好）"""
        return np.all(a >= b) and np.any(a > b)
    
    def _crowding_distance(self, fitness: np.ndarray, front: List[int]) -> np.ndarray:
        """
        计算拥挤度距离（用于保持多样性）
        """
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        
        for m in range(self.n_obj):
            # 按第m个目标排序
            sorted_indices = np.argsort(fitness[front, m])
            
            # 边界点设为无穷大
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # 计算中间点的拥挤度
            f_max = fitness[front[sorted_indices[-1]], m]
            f_min = fitness[front[sorted_indices[0]], m]
            
            if f_max - f_min > 1e-10:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        fitness[front[sorted_indices[i + 1]], m] - 
                        fitness[front[sorted_indices[i - 1]], m]
                    ) / (f_max - f_min)
        
        return distances

    def _select(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        选择操作：基于非支配排序和拥挤度
        """
        fronts = self._fast_non_dominated_sort(fitness)
        
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                new_population.extend(front)
            else:
                # 需要从当前前沿中选择部分
                remaining = self.pop_size - len(new_population)
                distances = self._crowding_distance(fitness, front)
                sorted_by_distance = np.argsort(distances)[::-1]  # 拥挤度大的优先
                new_population.extend([front[i] for i in sorted_by_distance[:remaining]])
                break
        
        return population[new_population]
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        交叉操作：两个父代方案交换部分地块
        """
        # 合并两个父代的地块
        all_sites = np.union1d(parent1, parent2)
        
        if len(all_sites) < 2 * self.n_select:
            # 地块不够，随机补充
            remaining = list(set(range(self.n_candidates)) - set(all_sites))
            if remaining:
                extra = np.random.choice(remaining, 
                                        min(len(remaining), 2 * self.n_select - len(all_sites)), 
                                        replace=False)
                all_sites = np.concatenate([all_sites, extra])
        
        # 随机分配给两个子代
        np.random.shuffle(all_sites)
        child1 = np.sort(all_sites[:self.n_select])
        child2 = np.sort(all_sites[self.n_select:2*self.n_select] if len(all_sites) >= 2*self.n_select 
                        else np.random.choice(self.n_candidates, self.n_select, replace=False))
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """
        变异操作：随机替换部分地块
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # 随机选择一个不在当前方案中的地块
                available = list(set(range(self.n_candidates)) - set(mutated))
                if available:
                    mutated[i] = random.choice(available)
        
        return np.sort(mutated)
