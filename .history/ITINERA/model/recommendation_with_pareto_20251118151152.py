"""
集成多目标优化的推荐示例
展示如何在实际推荐流程中使用帕累托前沿
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.multi_objective import MultiObjectiveOptimizer
from model.site_selector import SiteSelector
from model.utils.proxy_call import OpenaiCall


def recommend_with_pareto(user_reqs: str, 
                          top_k: int = 10,
                          use_pareto: bool = True):
    """
    使用多目标优化的推荐流程
    
    Args:
        user_reqs: 用户需求文本
        top_k: 返回地块数量
        use_pareto: 是否使用帕累托优化
    
    Returns:
        recommendation: 推荐结果
    """
    
    print("=" * 60)
    print("智能选址推荐（多目标优化版）")
    print("=" * 60)
    print(f"\n用户需求: {user_reqs}")
    print(f"返回数量: {top_k}")
    print(f"多目标优化: {'启用' if use_pareto else '禁用'}")
    print()
    
    # Step 1: 初始化选址系统
    print("Step 1: 初始化系统...")
    proxy = OpenaiCall()
    dataset_path = os.path.join(
        os.path.dirname(__file__), 
        'data', 
        'land_transactions_with_coordinates_metrics.csv'
    )
    
    selector = SiteSelector(
        user_reqs=user_reqs,
        min_site_candidate_num=top_k * 3,  # 候选数量为返回数量的3倍
        proxy_call=proxy,
        city='guangzhou',
        type='zh',
        dataset_path=dataset_path,
        enable_spatial_optimization=False  # 禁用空间优化，使用多目标优化
    )
    
    # Step 2: 语义检索候选地块
    print("\nStep 2: 语义检索候选地块...")
    req_topk_sites, pseudo_must_see = selector.get_candidate_sites()
    candidate_ids = req_topk_sites[:, 0].astype(int).tolist()
    print(f"✓ 检索到 {len(candidate_ids)} 个候选地块")
    
    # Step 3: 多目标优化（可选）
    if use_pareto:
        print("\nStep 3: 多目标优化（帕累托前沿）...")
        
        # 定义优化目标
        objectives = [
            {'name': '交通_便利评分(0-10)', 'maximize': True},
            {'name': '价格_万元/㎡', 'maximize': False},  # 价格越低越好
            {'name': '宗地面积(平方米)', 'maximize': True}
        ]
        
        # 创建优化器
        optimizer = MultiObjectiveOptimizer(selector.site_data)
        
        # 计算帕累托前沿
        pareto_indices = optimizer.pareto_front(objectives, candidate_ids)
        print(f"✓ 帕累托前沿包含 {len(pareto_indices)} 个地块")
        
        # 解释帕累托前沿
        explanation = optimizer.explain_pareto(pareto_indices, objectives)
        print(f"  多样性指标: {explanation['diversity']:.2f}")
        print("  权衡分析:")
        for trade_off in explanation['trade_offs']:
            print(f"    - {trade_off['objective']}最优: 地块{trade_off['best_index']} "
                  f"({trade_off['best_value']:.2f})")
        
        # 对帕累托前沿排序
        weights = selector.derive_scoring_weights()
        # 将权重映射到目标名称
        obj_weights = {
            '交通_便利评分(0-10)': weights.get('traffic', 0.34),
            '价格_万元/㎡': weights.get('price', 0.33),
            '宗地面积(平方米)': 0.1  # 面积权重较低
        }
        ranked = optimizer.rank_pareto_front(pareto_indices, objectives, obj_weights)
        
        # 取Top-K
        final_ids = [idx for idx, score in ranked[:top_k]]
        final_scores = [score for idx, score in ranked[:top_k]]
        
        print(f"\n✓ 从帕累托前沿中选择Top-{len(final_ids)}地块")
        
    else:
        print("\nStep 3: 简单排序...")
        # 使用原始评分排序
        final_ids = candidate_ids[:top_k]
        final_scores = req_topk_sites[:top_k, 1].tolist()
        print(f"✓ 按语义相似度选择Top-{len(final_ids)}地块")
    
    # Step 4: 生成推荐报告
    print("\nStep 4: 生成推荐报告...")
    
    # 构建推荐结果
    recommendations = []
    for i, site_id in enumerate(final_ids):
        row = selector.site_data.loc[site_id]
        
        # 计算综合评分
        weights = selector.derive_scoring_weights()
        composite = selector.composite_score(site_id, weights)
        
        recommendations.append({
            'rank': i + 1,
            'id': int(site_id),
            'name': row['宗地坐落'],
            'usage': row['土地用途'],
            'area': float(row['宗地面积(平方米)']),
            'price_total': float(row['挂牌起始价(万元)']),
            'price_unit': float(row['价格_万元/㎡']),
            'traffic_score': float(row['交通_便利评分(0-10)']),
            'composite_score': float(composite),
            'final_score': float(final_scores[i]) if i < len(final_scores) else 0.0,
            'lon': float(row['lon']),
            'lat': float(row['lat']),
            'context': row['context']
        })
    
    print(f"✓ 生成 {len(recommendations)} 条推荐")
    
    # Step 5: 展示结果
    print("\n" + "=" * 60)
    print("推荐结果")
    print("=" * 60)
    
    for rec in recommendations:
        print(f"\n【{rec['rank']}】{rec['name']}")
        print(f"  用途: {rec['usage']}")
        print(f"  面积: {rec['area']:.0f}㎡")
        print(f"  总价: {rec['price_total']:.0f}万元")
        print(f"  单价: {rec['price_unit']:.2f}万元/㎡")
        print(f"  交通: {rec['traffic_score']:.1f}分")
        print(f"  综合: {rec['composite_score']:.2f}分")
        if use_pareto:
            print(f"  帕累托分: {rec['final_score']:.2f}")
    
    return {
        'recommendations': recommendations,
        'use_pareto': use_pareto,
        'total': len(recommendations)
    }


def compare_with_without_pareto(user_reqs: str, top_k: int = 5):
    """
    对比使用和不使用多目标优化的结果
    """
    print("\n" + "=" * 80)
    print("对比测试：多目标优化 vs 简单排序")
    print("=" * 80)
    
    # 不使用多目标优化
    print("\n【方案A】简单排序（仅语义相似度）")
    result_simple = recommend_with_pareto(user_reqs, top_k, use_pareto=False)
    
    # 使用多目标优化
    print("\n\n【方案B】多目标优化（帕累托前沿）")
    result_pareto = recommend_with_pareto(user_reqs, top_k, use_pareto=True)
    
    # 对比分析
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    
    simple_ids = set(r['id'] for r in result_simple['recommendations'])
    pareto_ids = set(r['id'] for r in result_pareto['recommendations'])
    
    common = simple_ids & pareto_ids
    only_simple = simple_ids - pareto_ids
    only_pareto = pareto_ids - simple_ids
    
    print(f"\n相同地块: {len(common)}/{top_k}")
    print(f"仅简单排序: {len(only_simple)}")
    print(f"仅帕累托: {len(only_pareto)}")
    
    if only_pareto:
        print("\n帕累托优化新增的地块（更平衡的选择）:")
        for rec in result_pareto['recommendations']:
            if rec['id'] in only_pareto:
                print(f"  - {rec['name']}")
                print(f"    交通{rec['traffic_score']:.1f}分, "
                      f"价格{rec['price_unit']:.2f}万/㎡, "
                      f"面积{rec['area']:.0f}㎡")


if __name__ == '__main__':
    # 测试1: 单次推荐
    print("\n测试1: 单次推荐（使用多目标优化）")
    recommend_with_pareto(
        user_reqs="花都区食品生产厂用地，交通便利，价格便宜",
        top_k=5,
        use_pareto=True
    )
    
    # 测试2: 对比测试
    print("\n\n测试2: 对比测试")
    compare_with_without_pareto(
        user_reqs="花都区工业用地，面积大，交通便利",
        top_k=5
    )
