# encoding:utf-8
"""
使用百度地图圆形区域检索API获取地块周边POI数据
用于更新地块的交通、生活服务等指标

POI类型选择原则：
1. 交通设施 - 地铁站、公交站、停车场、火车站、高速出入口
2. 生活服务 - 银行、超市、医院、学校
3. 产业配套 - 物流园区、工业园区
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.parse
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# 百度地图API配置
BAIDU_AK = "tzB4hsfIx12cgWkdWtlq0YqCwhCjr34Q"
BAIDU_SK = "9ud02lmX54k2ROXUvYH3PE5TfjvepKXX"
BAIDU_API_HOST = "https://api.map.baidu.com"
BAIDU_AROUND_URI = "/place/v3/around"

# POI检索配置
POI_CONFIG = {
    # 交通设施
    "地铁站": {"query": "地铁站", "radius": 1500, "field_prefix": "交通_地铁"},
    "公交站": {"query": "公交站", "radius": 500, "field_prefix": "交通_公交"},
    "停车场": {"query": "停车场", "radius": 1000, "field_prefix": "交通_停车"},
    "火车站": {"query": "火车站", "radius": 3000, "field_prefix": "交通_火车"},
    "高速出入口": {"query": "高速出入口$高速公路出口$高速公路入口", "radius": 5000, "field_prefix": "交通_高速"},
    
    # 生活服务
    "银行": {"query": "银行", "radius": 1000, "field_prefix": "生活_银行"},
    "超市": {"query": "超市$便利店", "radius": 1000, "field_prefix": "生活_超市"},
    "医院": {"query": "医院$诊所", "radius": 2000, "field_prefix": "生活_医院"},
    "餐饮": {"query": "餐厅$快餐", "radius": 500, "field_prefix": "生活_餐饮"},
    
    # 教育
    "学校": {"query": "学校$小学$中学", "radius": 2000, "field_prefix": "教育_学校"},
    
    # 产业配套
    "物流园": {"query": "物流园$物流中心$快递", "radius": 3000, "field_prefix": "产业_物流"},
    "工业园": {"query": "工业园$产业园$科技园", "radius": 3000, "field_prefix": "产业_园区"},
}


def calculate_sn(uri: str, params: dict, sk: str) -> str:
    """计算百度地图API的sn签名"""
    params_arr = [f"{key}={params[key]}" for key in params]
    query_str = uri + "?" + "&".join(params_arr)
    encoded_str = urllib.parse.quote(query_str, safe="/:=&?#+!$,;'@()*[]")
    raw_str = encoded_str + sk
    sn = hashlib.md5(urllib.parse.quote_plus(raw_str).encode("utf8")).hexdigest()
    return sn


def search_around_poi(lat: float, lon: float, query: str, radius: int, 
                      page_num: int = 0, page_size: int = 20) -> dict:
    """
    圆形区域检索POI
    
    Args:
        lat: 纬度 (WGS84)
        lon: 经度 (WGS84)
        query: 检索关键字，多个用$分隔
        radius: 检索半径（米）
        page_num: 页码
        page_size: 每页数量
    
    Returns:
        API响应JSON
    """
    # 注意：百度地图默认使用bd09ll坐标，需要指定coord_type=1表示WGS84
    params = {
        "query": query,
        "location": f"{lat},{lon}",
        "radius": str(radius),
        "radius_limit": "true",  # 严格限制在半径内
        "coord_type": "1",  # WGS84坐标
        "output": "json",
        "scope": "2",  # 返回详细信息（包含距离）
        "page_num": str(page_num),
        "page_size": str(page_size),
        "ak": BAIDU_AK,
    }
    
    sn = calculate_sn(BAIDU_AROUND_URI, params, BAIDU_SK)
    params["sn"] = sn
    
    url = BAIDU_API_HOST + BAIDU_AROUND_URI
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[API错误] {e}")
        return {"status": -1, "message": str(e), "results": []}


def get_poi_stats(lat: float, lon: float, query: str, radius: int) -> Tuple[int, Optional[float]]:
    """
    获取指定位置周边POI的统计信息
    
    Returns:
        (数量, 最近距离)
    """
    result = search_around_poi(lat, lon, query, radius)
    
    if result.get("status") != 0:
        print(f"  [警告] API返回错误: {result.get('message', 'unknown')}")
        return 0, None
    
    pois = result.get("results", [])
    count = len(pois)
    
    # 获取最近距离
    min_distance = None
    for poi in pois:
        detail_info = poi.get("detail_info", {})
        distance = detail_info.get("distance")
        if distance is not None:
            distance = float(distance)
            if min_distance is None or distance < min_distance:
                min_distance = distance
    
    return count, min_distance


def process_single_site(idx: int, row: pd.Series, poi_config: dict) -> dict:
    """处理单个地块的POI数据"""
    lat = float(row["lat"])
    lon = float(row["lon"])
    name = str(row.get("宗地坐落", row.get("name", f"地块{idx}")))[:30]
    
    print(f"\n[{idx+1}] 处理: {name}")
    print(f"    坐标: ({lat:.6f}, {lon:.6f})")
    
    result = {"index": idx}
    
    for poi_name, config in poi_config.items():
        query = config["query"]
        radius = config["radius"]
        field_prefix = config["field_prefix"]
        
        count, min_dist = get_poi_stats(lat, lon, query, radius)
        
        result[f"{field_prefix}数量({radius/1000:.1f}km)"] = count
        result[f"{field_prefix}最近距离(m)"] = min_dist if min_dist else None
        
        dist_str = f"{min_dist:.0f}m" if min_dist else "无"
        print(f"    {poi_name}: {count}个 (半径{radius}m), 最近{dist_str}")
        
        # API限流：每秒最多5次请求
        time.sleep(0.25)
    
    return result


def calculate_traffic_score(row: dict) -> float:
    """
    计算交通便利性综合评分 (0-10分)
    
    评分规则：
    - 地铁站(1.5km): 有=+3分, 距离<500m额外+1分
    - 公交站(0.5km): 有=+2分, >=3个额外+1分
    - 停车场(1km): 有=+1分, >=5个额外+0.5分
    - 火车站(3km): 有=+1分
    - 高速出入口(5km): 有=+1分, 距离<2km额外+0.5分
    """
    score = 0.0
    
    # 地铁站
    subway_count = row.get("交通_地铁数量(1.5km)", 0) or 0
    subway_dist = row.get("交通_地铁最近距离(m)")
    if subway_count > 0:
        score += 3.0
        if subway_dist and subway_dist < 500:
            score += 1.0
    
    # 公交站
    bus_count = row.get("交通_公交数量(0.5km)", 0) or 0
    if bus_count > 0:
        score += 2.0
        if bus_count >= 3:
            score += 1.0
    
    # 停车场
    parking_count = row.get("交通_停车数量(1.0km)", 0) or 0
    if parking_count > 0:
        score += 1.0
        if parking_count >= 5:
            score += 0.5
    
    # 火车站
    train_count = row.get("交通_火车数量(3.0km)", 0) or 0
    if train_count > 0:
        score += 1.0
    
    # 高速出入口
    highway_count = row.get("交通_高速数量(5.0km)", 0) or 0
    highway_dist = row.get("交通_高速最近距离(m)")
    if highway_count > 0:
        score += 1.0
        if highway_dist and highway_dist < 2000:
            score += 0.5
    
    return min(score, 10.0)


def main():
    """主函数"""
    # 数据文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(data_dir, "land_transactions_with_coordinates_metrics.csv")
    output_path = os.path.join(data_dir, "land_transactions_with_coordinates_metrics_new.csv")
    
    print("=" * 60)
    print("百度地图POI数据获取工具")
    print("=" * 60)
    print(f"输入文件: {csv_path}")
    print(f"输出文件: {output_path}")
    print(f"POI类型: {list(POI_CONFIG.keys())}")
    print("=" * 60)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    print(f"\n共 {len(df)} 个地块待处理")
    
    # 处理每个地块
    all_results = []
    for idx, row in df.iterrows():
        result = process_single_site(idx, row, POI_CONFIG)
        all_results.append(result)
    
    # 合并结果
    poi_df = pd.DataFrame(all_results)
    poi_df.set_index("index", inplace=True)
    
    # 计算交通便利性评分
    poi_df["交通_便利评分(0-10)_new"] = poi_df.apply(calculate_traffic_score, axis=1)
    
    # 更新原数据
    # 删除旧的交通列
    old_cols = [c for c in df.columns if c.startswith("交通_")]
    df = df.drop(columns=old_cols, errors='ignore')
    
    # 合并新数据
    df = df.join(poi_df)
    
    # 重命名评分列
    if "交通_便利评分(0-10)_new" in df.columns:
        df = df.rename(columns={"交通_便利评分(0-10)_new": "交通_便利评分(0-10)"})
    
    # 保存
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 数据已保存到: {output_path}")
    
    # 打印统计
    print("\n" + "=" * 60)
    print("POI统计摘要")
    print("=" * 60)
    for col in poi_df.columns:
        if "数量" in col:
            mean_val = poi_df[col].mean()
            max_val = poi_df[col].max()
            print(f"{col}: 平均{mean_val:.1f}, 最大{max_val}")
    
    print(f"\n交通便利评分: 平均{poi_df['交通_便利评分(0-10)_new'].mean():.2f}, "
          f"最高{poi_df['交通_便利评分(0-10)_new'].max():.2f}")


if __name__ == "__main__":
    main()
