# encoding:utf-8
"""
百度地图圆形区域检索API - 地块周边POI数据获取工具 v2

修复问题：
1. 正确处理分页获取所有POI
2. 严格使用radius_limit限制检索范围
3. 使用detail_info中的distance字段验证距离
4. 增加更多对选址有价值的POI类型

POI类型选择原则（用于空间选址推荐）：
1. 交通设施 - 影响物流成本和员工通勤
2. 生活配套 - 影响员工生活便利性
3. 教育资源 - 影响人才招聘
4. 产业配套 - 影响供应链和协作
5. 商业环境 - 影响商务活动
"""

import os
import sys
import json
import time
import hashlib
import urllib.parse
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# 百度地图API配置
# ============================================================
BAIDU_AK = "tzB4hsfIx12cgWkdWtlq0YqCwhCjr34Q"
BAIDU_SK = "9ud02lmX54k2ROXUvYH3PE5TfjvepKXX"
BAIDU_API_HOST = "https://api.map.baidu.com"
BAIDU_AROUND_URI = "/place/v2/search"  # 使用v2版本，更稳定

# API限流配置
API_RATE_LIMIT = 0.3  # 每次请求间隔（秒）
MAX_RETRIES = 3       # 最大重试次数

# ============================================================
# POI检索配置 - 根据选址需求精选
# ============================================================
POI_CONFIG = {
    # === 交通设施 ===
    "地铁站": {
        "query": "地铁站",
        "radius": 1500,
        "category": "交通",
        "weight": 3.0,  # 评分权重
        "description": "轨道交通，员工通勤首选"
    },
    "公交站": {
        "query": "公交站",
        "radius": 500,
        "category": "交通",
        "weight": 2.0,
        "description": "公共交通覆盖"
    },
    "停车场": {
        "query": "停车场",
        "radius": 1000,
        "category": "交通",
        "weight": 1.0,
        "description": "车辆停放便利性"
    },
    "火车站": {
        "query": "火车站",
        "radius": 5000,
        "category": "交通",
        "weight": 1.5,
        "description": "长途交通枢纽"
    },
    "高速出入口": {
        "query": "高速公路出入口",
        "radius": 5000,
        "category": "交通",
        "weight": 2.0,
        "description": "物流运输便利性"
    },
    
    # === 生活配套 ===
    "银行": {
        "query": "银行",
        "radius": 1000,
        "category": "生活",
        "weight": 1.0,
        "description": "金融服务"
    },
    "超市便利店": {
        "query": "超市",
        "radius": 1000,
        "category": "生活",
        "weight": 1.5,
        "description": "日常购物"
    },
    "餐饮": {
        "query": "餐厅",
        "radius": 500,
        "category": "生活",
        "weight": 1.5,
        "description": "员工就餐"
    },
    "医院": {
        "query": "医院",
        "radius": 3000,
        "category": "生活",
        "weight": 2.0,
        "description": "医疗保障"
    },
    "药店": {
        "query": "药店",
        "radius": 1000,
        "category": "生活",
        "weight": 0.5,
        "description": "基础医疗"
    },
    
    # === 教育资源 ===
    "高校": {
        "query": "大学",
        "radius": 5000,
        "category": "教育",
        "weight": 2.0,
        "description": "人才来源"
    },
    "中小学": {
        "query": "学校",
        "radius": 2000,
        "category": "教育",
        "weight": 1.0,
        "description": "员工子女教育"
    },
    
    # === 产业配套 ===
    "物流园": {
        "query": "物流园区",
        "radius": 5000,
        "category": "产业",
        "weight": 2.0,
        "description": "物流配套"
    },
    "工业园": {
        "query": "工业园区",
        "radius": 5000,
        "category": "产业",
        "weight": 1.5,
        "description": "产业集聚"
    },
    
    # === 商业环境 ===
    "写字楼": {
        "query": "写字楼",
        "radius": 2000,
        "category": "商业",
        "weight": 1.0,
        "description": "商务环境"
    },
    "酒店": {
        "query": "酒店",
        "radius": 2000,
        "category": "商业",
        "weight": 1.0,
        "description": "商务接待"
    },
}


def calculate_sn(uri: str, params: dict, sk: str) -> str:
    """计算百度地图API的sn签名"""
    # 按key排序
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    params_str = "&".join([f"{k}={v}" for k, v in sorted_params])
    query_str = uri + "?" + params_str
    
    # URL编码
    encoded_str = urllib.parse.quote(query_str, safe="/:=&?#+!$,;'@()*[]")
    raw_str = encoded_str + sk
    
    # MD5签名
    sn = hashlib.md5(urllib.parse.quote_plus(raw_str).encode("utf8")).hexdigest()
    return sn


def search_around_poi(
    lat: float, 
    lon: float, 
    query: str, 
    radius: int,
    page_num: int = 0,
    page_size: int = 20
) -> dict:
    """
    圆形区域检索POI（单页）
    
    Args:
        lat: 纬度 (WGS84)
        lon: 经度 (WGS84)  
        query: 检索关键字
        radius: 检索半径（米）
        page_num: 页码（从0开始）
        page_size: 每页数量（最大20）
    
    Returns:
        API响应JSON
    """
    params = {
        "query": query,
        "location": f"{lat},{lon}",
        "radius": str(radius),
        "radius_limit": "true",      # 严格限制在半径内
        "coord_type": "1",           # 1=WGS84坐标
        "ret_coordtype": "gcj02ll",  # 返回国测局坐标
        "output": "json",
        "scope": "2",                # 返回详细信息
        "page_num": str(page_num),
        "page_size": str(page_size),
        "ak": BAIDU_AK,
    }
    
    # 计算签名
    sn = calculate_sn(BAIDU_AROUND_URI, params, BAIDU_SK)
    params["sn"] = sn
    
    url = BAIDU_API_HOST + BAIDU_AROUND_URI
    
    for retry in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == 0:
                return result
            elif result.get("status") == 302:
                # 配额超限，等待后重试
                print(f"    [限流] 等待重试...")
                time.sleep(2)
                continue
            else:
                print(f"    [API错误] status={result.get('status')}, msg={result.get('message')}")
                return {"status": result.get("status"), "results": [], "total": 0}
                
        except requests.exceptions.Timeout:
            print(f"    [超时] 重试 {retry+1}/{MAX_RETRIES}")
            time.sleep(1)
        except Exception as e:
            print(f"    [异常] {e}")
            return {"status": -1, "results": [], "total": 0}
    
    return {"status": -1, "results": [], "total": 0}


def get_all_pois_in_radius(
    lat: float, 
    lon: float, 
    query: str, 
    radius: int,
    max_pages: int = 5
) -> List[dict]:
    """
    获取指定半径内的所有POI（自动分页）
    
    Args:
        lat, lon: 中心点坐标
        query: 检索关键字
        radius: 检索半径
        max_pages: 最大页数（防止无限循环）
    
    Returns:
        POI列表
    """
    all_pois = []
    page_num = 0
    page_size = 20
    
    while page_num < max_pages:
        result = search_around_poi(lat, lon, query, radius, page_num, page_size)
        
        if result.get("status") != 0:
            break
            
        pois = result.get("results", [])
        if not pois:
            break
        
        # 过滤：只保留在半径内的POI（使用distance字段二次验证）
        for poi in pois:
            detail_info = poi.get("detail_info", {})
            distance = detail_info.get("distance")
            
            # 如果有距离信息，验证是否在半径内
            if distance is not None:
                try:
                    dist_val = float(distance)
                    if dist_val <= radius:
                        poi["_distance"] = dist_val
                        all_pois.append(poi)
                except:
                    all_pois.append(poi)
            else:
                all_pois.append(poi)
        
        # 检查是否还有更多
        total = result.get("total", 0)
        if len(all_pois) >= total or len(pois) < page_size:
            break
            
        page_num += 1
        time.sleep(API_RATE_LIMIT)
    
    return all_pois
