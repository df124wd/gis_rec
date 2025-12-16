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

    def optimize(self) -> Dict:
        """
        运行NSGA-II优化
        
        Returns:
            {
                'pareto_front': 帕累托前沿方案列表,
                'pareto_fitness': 对应的目标值,
                'best_solutions': 各目标最优的方案,
                'generations': 实际进化代数
            }
        """
        print(f"\n[NSGA-II] 开始优化...")
        print(f"  候选地块: {self.n_candidates}个")
        print(f"  每方案选择: {self.n_select}个地块")
        print(f"  种群大小: {self.pop_size}")
        print(f"  进化代数: {self.n_generations}")
        
        # 初始化种群
        population = self._init_population()
        fitness = self._evaluate(population)
        
        # 进化循环
        for gen in range(self.n_generations):
            # 生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                # 锦标赛选择父代
                idx1, idx2 = random.sample(range(len(population)), 2)
                parent1, parent2 = population[idx1], population[idx2]
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                offspring.extend([child1, child2])
            
            offspring = np.array(offspring[:self.pop_size])
            offspring_fitness = self._evaluate(offspring)
            
            # 合并父代和子代
            combined_pop = np.vstack([population, offspring])
            combined_fitness = np.vstack([fitness, offspring_fitness])
            
            # 选择下一代
            population = self._select(combined_pop, combined_fitness)
            fitness = self._evaluate(population)
            
            # 每10代打印进度
            if (gen + 1) % 10 == 0:
                fronts = self._fast_non_dominated_sort(fitness)
                print(f"  第{gen+1}代: 帕累托前沿{len(fronts[0])}个方案")
        
        # 提取最终帕累托前沿
        fronts = self._fast_non_dominated_sort(fitness)
        pareto_indices = fronts[0]
        
        pareto_front = population[pareto_indices]
        pareto_fitness = fitness[pareto_indices]
        
        # 找出各目标最优的方案
        best_solutions = {}
        for j, obj in enumerate(self.objectives):
            best_idx = np.argmax(pareto_fitness[:, j])
            best_solutions[obj.get('weight_key', obj['name'])] = {
                'solution_idx': pareto_indices[best_idx],
                'sites': pareto_front[best_idx].tolist(),
                'fitness': pareto_fitness[best_idx].tolist(),
                'best_value': pareto_fitness[best_idx, j]
            }
        
        print(f"\n[NSGA-II] 优化完成!")
        print(f"  帕累托前沿: {len(pareto_front)}个方案")
        
        return {
            'pareto_front': pareto_front,
            'pareto_fitness': pareto_fitness,
            'best_solutions': best_solutions,
            'generations': self.n_generations
        }

    def select_best_solution(self, pareto_front: np.ndarray, 
                             pareto_fitness: np.ndarray,
                             weights: Dict[str, float]) -> Tuple[np.ndarray, int]:
        """
        根据用户偏好从帕累托前沿中选择最佳方案
        
        Args:
            pareto_front: 帕累托前沿方案
            pareto_fitness: 对应的目标值
            weights: 用户偏好权重
        
        Returns:
            best_solution: 最佳方案（地块索引列表）
            best_idx: 在帕累托前沿中的索引
        """
        # 构建权重向量
        weight_vector = np.zeros(self.n_obj)
        for j, obj in enumerate(self.objectives):
            key = obj.get('weight_key', obj['name'])
            weight_vector[j] = weights.get(key, 0.25)
        
        # 归一化权重
        weight_vector = weight_vector / weight_vector.sum()
        
        # 计算加权得分
        scores = np.dot(pareto_fitness, weight_vector)
        
        best_idx = np.argmax(scores)
        return pareto_front[best_idx], best_idx
    
    def get_solution_details(self, solution: np.ndarray) -> List[Dict]:
        """
        获取方案中每个地块的详细信息
        """
        details = []
        for idx in solution:
            row = self.data.iloc[idx]
            detail = {
                'index': int(idx),
                'name': str(row.get('宗地坐落', f'地块{idx}')),
                'objectives': {}
            }
            for j, obj in enumerate(self.objectives):
                col = obj['name']
                key = obj.get('weight_key', col)
                detail['objectives'][key] = {
                    'raw': float(self.obj_raw[idx, j]),
                    'normalized': float(self.obj_values[idx, j])
                }
            details.append(detail)
        return details


class HardConstraintFilter:
    """硬约束过滤器"""
    
    @staticmethod
    def apply(data: pd.DataFrame, constraints: List[Dict]) -> pd.DataFrame:
        """应用硬约束过滤"""
        filtered = data.copy()
        
        for c in constraints:
            field = c.get('field')
            op = c.get('operator')
            value = c.get('value')
            
            if field not in filtered.columns:
                print(f"  [跳过] 字段不存在: {field}")
                continue
            
            original_count = len(filtered)
            
            try:
                if op == 'contains':
                    mask = filtered[field].astype(str).str.contains(str(value), na=False)
                    filtered = filtered[mask]
                elif op == 'not_contains':
                    mask = ~filtered[field].astype(str).str.contains(str(value), na=False)
                    filtered = filtered[mask]
                elif op == '>=':
                    col = pd.to_numeric(filtered[field], errors='coerce')
                    filtered = filtered[col >= float(value)]
                elif op == '<=':
                    col = pd.to_numeric(filtered[field], errors='coerce')
                    filtered = filtered[col <= float(value)]
                
                print(f"  [约束] {c.get('original_text', field)}: {original_count} → {len(filtered)}")
            except Exception as e:
                print(f"  [错误] {field}: {e}")
        
        return filtered.reset_index(drop=True)


def run_nsga2_optimization(candidate_data: pd.DataFrame,
                           objectives: List[Dict],
                           weights: Dict[str, float],
                           constraints: List[Dict] = None,
                           n_select: int = 10,
                           pop_size: int = 100,
                           n_generations: int = 50,
                           top_k: int = 5) -> Dict:
    """
    运行完整的NSGA-II多目标优化流程
    
    Args:
        candidate_data: 候选地块数据
        objectives: 优化目标
        weights: 用户偏好权重
        constraints: 硬约束
        n_select: 每方案选择地块数
        pop_size: 种群大小
        n_generations: 进化代数
        top_k: 最终推荐数量
    
    Returns:
        {
            'recommended_sites': 推荐地块列表,
            'pareto_info': 帕累托前沿信息,
            'selected_solution': 选中的方案信息
        }
    """
    print("\n" + "="*70)
    print("[NSGA-II 多目标优化] 完整版")
    print("="*70)
    
    # 1. 硬约束过滤
    if constraints:
        print(f"\n[阶段1] 硬约束过滤")
        print(f"  原始候选: {len(candidate_data)}个")
        filtered_data = HardConstraintFilter.apply(candidate_data, constraints)
        print(f"  过滤后: {len(filtered_data)}个")
        
        if len(filtered_data) == 0:
            print("  [警告] 过滤后无候选，放宽约束...")
            filtered_data = candidate_data.copy().reset_index(drop=True)
    else:
        filtered_data = candidate_data.copy().reset_index(drop=True)
        print(f"\n[阶段1] 无硬约束，使用全部{len(filtered_data)}个候选")
    
    # 保存原始索引映射
    if '_original_idx' not in filtered_data.columns:
        filtered_data['_original_idx'] = filtered_data.index
    
    # 调整n_select
    actual_n_select = min(n_select, len(filtered_data))
    if actual_n_select < n_select:
        print(f"  [调整] 候选不足，每方案选择{actual_n_select}个地块")
    
    # 2. NSGA-II优化
    print(f"\n[阶段2] NSGA-II遗传算法优化")
    
    optimizer = NSGA2Optimizer(
        candidate_data=filtered_data,
        objectives=objectives,
        n_select=actual_n_select,
        pop_size=pop_size,
        n_generations=n_generations
    )
    
    result = optimizer.optimize()
    
    pareto_front = result['pareto_front']
    pareto_fitness = result['pareto_fitness']
    
    # 3. 根据用户偏好选择最佳方案
    print(f"\n[阶段3] 根据用户偏好选择方案")
    print(f"  权重: {weights}")
    
    best_solution, best_idx = optimizer.select_best_solution(
        pareto_front, pareto_fitness, weights
    )
    
    print(f"  选中方案{best_idx}: 包含{len(best_solution)}个地块")
    
    # 打印各目标最优方案
    print(f"\n[帕累托前沿分析]")
    for key, info in result['best_solutions'].items():
        print(f"  {key}最优: 方案包含地块{info['sites'][:3]}... (值={info['best_value']:.3f})")
