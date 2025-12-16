# encoding:utf-8
"""
高德地图周边搜索API - 地块周边POI数据获取工具 v2

功能：
1. 获取地块周边各类POI数据
2. 计算各维度评分（交通、生活、教育、产业、商业）
3. 输出POI详情JSON文件供前端展示

高德地图周边搜索API文档：
https://lbs.amap.com/api/webservice/guide/api/newpoisearch

使用方法：
    python fetch_poi_metrics_v2.py
"""

import os
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# 高德地图API配置
# ============================================================
AMAP_KEY = "52fe59f6fa4c4000739004426942af84"
AMAP_API_URL = "https://restapi.amap.com/v5/place/around"

# API限流配置
API_RATE_LIMIT = 0.3      # 每次API请求间隔（秒）
SITE_INTERVAL = 2.0       # 每个地块处理完后等待（秒）
MAX_RETRIES = 3

# ============================================================
# POI类型配置
# 高德地图POI分类编码参考：https://lbs.amap.com/api/webservice/download
# ============================================================
POI_TYPES = [
    # (类型名, POI类型编码, 半径, 类别)
    # 交通设施
    ("地铁站", "150500", 1500, "交通"),      # 地铁站
    ("公交站", "150700", 500, "交通"),       # 公交站
    ("停车场", "150900", 1000, "交通"),      # 停车场
    ("火车站", "150200", 5000, "交通"),      # 火车站
    ("高速出入口", "150302", 5000, "交通"),  # 高速公路出入口
    # 生活服务
    ("银行", "160100", 1000, "生活"),        # 银行
    ("超市", "060400", 1000, "生活"),        # 超市
    ("餐饮", "050000", 500, "生活"),         # 餐饮服务
    ("医院", "090100", 3000, "生活"),        # 综合医院
    ("药店", "090600", 1000, "生活"),        # 药店
    # 教育
    ("高校", "141200", 5000, "教育"),        # 高等院校
    ("中小学", "141203", 2000, "教育"),      # 中学 (也可用141200|141201|141202)
    # 产业
    ("物流园", "170200", 5000, "产业"),      # 物流速递
    ("工业园", "170100", 5000, "产业"),      # 工厂
    # 商业
    ("写字楼", "120200", 2000, "商业"),      # 商务住宅-楼宇
    ("酒店", "100000", 2000, "商业"),        # 住宿服务
]


def search_around_poi(lon: float, lat: float, poi_type: str, radius: int,
                      page_num: int = 1, page_size: int = 25) -> dict:
    """
    高德地图周边搜索API
    
    Args:
        lon: 经度 (WGS84，高德会自动转换)
        lat: 纬度
        poi_type: POI类型编码
        radius: 搜索半径（米），最大50000
        page_num: 页码，从1开始
        page_size: 每页数量，最大25
    
    Returns:
        API响应JSON
    """
    params = {
        "key": AMAP_KEY,
        "location": f"{lon},{lat}",
        "types": poi_type,
        "radius": str(radius),
        "sortrule": "distance",  # 按距离排序
        "page_num": str(page_num),
        "page_size": str(page_size),
        "show_fields": "business,distance",  # 返回额外字段
    }
    
    for retry in range(MAX_RETRIES):
        try:
            response = requests.get(AMAP_API_URL, params=params, timeout=15)
            result = response.json()
            
            if result.get("status") == "1":
                return result
            else:
                info = result.get("info", "")
                infocode = result.get("infocode", "")
                if infocode == "10003":  # 访问已超出日访问量
                    print(f"[配额超限]", end=" ")
                    return {"status": "0", "pois": [], "count": "0"}
                elif infocode == "10004":  # 访问过于频繁
                    print(f"[限流] 等待...", end=" ", flush=True)
                    time.sleep(2)
                    continue
                else:
                    print(f"[API错误:{infocode}:{info}]", end=" ")
                    return {"status": "0", "pois": [], "count": "0"}
        except Exception as e:
            print(f"[异常:{e}]", end=" ")
            time.sleep(1)
    
    return {"status": "0", "pois": [], "count": "0"}


def get_pois_with_details(lon: float, lat: float, poi_type: str, radius: int,
                          max_pages: int = 3) -> Tuple[int, Optional[float], List[dict]]:
    """
    获取POI统计和详情（自动分页）
    
    Returns:
        (数量, 最近距离, POI详情列表)
    """
    all_pois = []
    page_num = 1
    
    while page_num <= max_pages:
        result = search_around_poi(lon, lat, poi_type, radius, page_num)
        
        if result.get("status") != "1":
            break
        
        pois = result.get("pois", [])
        if not pois:
            break
        
        for poi in pois:
            # 提取有用信息
            distance = poi.get("distance")
            poi_info = {
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "location": poi.get("location", ""),  # "lon,lat"格式
                "distance": int(distance) if distance else None,
                "tel": poi.get("tel", ""),
                "type": poi.get("type", ""),
                "typecode": poi.get("typecode", ""),
            }
            
            # 验证距离在半径内
            if poi_info["distance"] is not None and poi_info["distance"] <= radius:
                all_pois.append(poi_info)
            elif poi_info["distance"] is None:
                all_pois.append(poi_info)
        
        # 检查是否还有更多
        total = int(result.get("count", 0))
        if len(all_pois) >= total or len(pois) < 25:
            break
        
        page_num += 1
        time.sleep(API_RATE_LIMIT)
    
    # 计算统计
    count = len(all_pois)
    min_dist = None
    for p in all_pois:
        if p["distance"] is not None:
            if min_dist is None or p["distance"] < min_dist:
                min_dist = p["distance"]
    
    return count, min_dist, all_pois


@dataclass
class SiteMetrics:
    """地块POI指标"""
    index: int
    name: str
    lat: float
    lon: float
    poi_data: Dict[str, dict] = field(default_factory=dict)
    
    # 评分
    traffic_score: float = 0.0
    life_score: float = 0.0
    education_score: float = 0.0
    industry_score: float = 0.0
    business_score: float = 0.0


def process_single_site(idx: int, row: pd.Series) -> SiteMetrics:
    """处理单个地块的POI数据"""
    lat = float(row["lat"])
    lon = float(row["lon"])
    name = str(row.get("宗地坐落", f"地块{idx}"))[:40]
    
    print(f"\n[{idx+1}] {name}")
    print(f"    坐标: ({lat:.6f}, {lon:.6f})")
    
    metrics = SiteMetrics(index=idx, name=name, lat=lat, lon=lon)
    
    current_category = None
    for poi_name, poi_type, radius, category in POI_TYPES:
        if category != current_category:
            if current_category is not None:
                print()
            print(f"    [{category}]", end=" ")
            current_category = category
        
        # 获取POI
        count, min_dist, pois = get_pois_with_details(lon, lat, poi_type, radius)
        
        metrics.poi_data[poi_name] = {
            "count": count,
            "min_distance": min_dist,
            "radius": radius,
            "pois": pois
        }
        
        print(f"{poi_name}:{count}", end=" ", flush=True)
        time.sleep(API_RATE_LIMIT)
    
    print()
    
    # 计算评分
    calculate_scores(metrics)
    print(f"    [评分] 交通:{metrics.traffic_score:.1f} 生活:{metrics.life_score:.1f} "
          f"教育:{metrics.education_score:.1f} 产业:{metrics.industry_score:.1f} "
          f"商业:{metrics.business_score:.1f}")
    
    return metrics
