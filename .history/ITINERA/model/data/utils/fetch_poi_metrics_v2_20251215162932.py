# encoding:utf-8
"""
百度地图圆形区域检索API - 地块周边POI数据获取工具 v2

功能：
1. 获取地块周边各类POI数据
2. 计算各维度评分（交通、生活、教育、产业、商业）
3. 输出POI详情JSON文件供前端展示
4. 应对个人认证API限流

使用方法：
    python fetch_poi_metrics_v2.py
"""

import os
import json
import time
import hashlib
import urllib.parse
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# 百度地图API配置
# ============================================================
BAIDU_AK = "tzB4hsfIx12cgWkdWtlq0YqCwhCjr34Q"
BAIDU_SK = "9ud02lmX54k2ROXUvYH3PE5TfjvepKXX"
BAIDU_API_HOST = "https://api.map.baidu.com"
BAIDU_AROUND_URI = "/place/v3/around"

# API限流配置（个人认证配额较少）
API_RATE_LIMIT = 0.5      # 每次API请求间隔（秒）
SITE_INTERVAL = 3.0       # 每个地块处理完后等待（秒）
RETRY_WAIT = 5.0          # 限流重试等待（秒）
MAX_RETRIES = 3


def calculate_sn(uri: str, params: dict, sk: str) -> Tuple[str, str]:
    """计算百度地图API的sn签名"""
    params_arr = [f"{key}={params[key]}" for key in params]
    query_str = uri + "?" + "&".join(params_arr)
    encoded_str = urllib.parse.quote(query_str, safe="/:=&?#+!$,;'@()*[]")
    raw_str = encoded_str + sk
    sn = hashlib.md5(urllib.parse.quote_plus(raw_str).encode("utf8")).hexdigest()
    return sn, query_str


def search_around_poi(lat: float, lon: float, query: str, radius: int,
                      page_num: int = 0, page_size: int = 20) -> dict:
    """圆形区域检索POI（单页）"""
    params = {
        "query": query,
        "location": f"{lat},{lon}",
        "radius": str(radius),
        "radius_limit": "true",
        "coord_type": "1",
        "output": "json",
        "scope": "2",
        "page_num": str(page_num),
        "page_size": str(page_size),
        "ak": BAIDU_AK,
    }
    
    sn, query_str = calculate_sn(BAIDU_AROUND_URI, params, BAIDU_SK)
    url = BAIDU_API_HOST + query_str + "&sn=" + sn
    
    for retry in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=15)
            result = response.json()
            
            if result.get("status") == 0:
                return result
            elif result.get("status") == 302:
                print(f"[限流] 等待{RETRY_WAIT}秒...", end=" ", flush=True)
                time.sleep(RETRY_WAIT)
                continue
            else:
                return {"status": result.get("status"), "results": [], "total": 0}
        except Exception as e:
            print(f"[异常:{e}]", end=" ")
            time.sleep(1)
    
    return {"status": -1, "results": [], "total": 0}


def get_pois_with_details(lat: float, lon: float, query: str, radius: int,
                          max_pages: int = 3) -> Tuple[int, Optional[float], List[dict]]:
    """
    获取POI统计和详情
    
    Returns:
        (数量, 最近距离, POI详情列表)
    """
    all_pois = []
    page_num = 0
    
    while page_num < max_pages:
        result = search_around_poi(lat, lon, query, radius, page_num)
        
        if result.get("status") != 0:
            break
        
        pois = result.get("results", [])
        if not pois:
            break
        
        for poi in pois:
            detail_info = poi.get("detail_info", {})
            distance = detail_info.get("distance")
            
            # 提取有用信息
            poi_info = {
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "location": poi.get("location", {}),
                "distance": float(distance) if distance else None,
                "telephone": poi.get("telephone", ""),
                "tag": detail_info.get("tag", ""),
            }
            
            if distance is not None:
                try:
                    if float(distance) <= radius:
                        all_pois.append(poi_info)
                except:
                    all_pois.append(poi_info)
            else:
                all_pois.append(poi_info)
        
        total = result.get("total", 0)
        if len(all_pois) >= total or len(pois) < 20:
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
    
    # 各类POI数据
    poi_data: Dict[str, dict] = field(default_factory=dict)
    
    # 评分
    traffic_score: float = 0.0
    life_score: float = 0.0
    education_score: float = 0.0
    industry_score: float = 0.0
    business_score: float = 0.0


# POI检索配置
POI_TYPES = [
    # (类型名, 查询词, 半径, 类别)
    ("地铁站", "地铁站", 1500, "交通"),
    ("公交站", "公交站", 500, "交通"),
    ("停车场", "停车场", 1000, "交通"),
    ("火车站", "火车站", 5000, "交通"),
    ("高速出入口", "高速公路出入口", 5000, "交通"),
    ("银行", "银行", 1000, "生活"),
    ("超市", "超市", 1000, "生活"),
    ("餐饮", "餐厅", 500, "生活"),
    ("医院", "医院", 3000, "生活"),
    ("药店", "药店", 1000, "生活"),
    ("高校", "大学", 5000, "教育"),
    ("中小学", "学校", 2000, "教育"),
    ("物流园", "物流园区", 5000, "产业"),
    ("工业园", "工业园区", 5000, "产业"),
    ("写字楼", "写字楼", 2000, "商业"),
    ("酒店", "酒店", 2000, "商业"),
]
