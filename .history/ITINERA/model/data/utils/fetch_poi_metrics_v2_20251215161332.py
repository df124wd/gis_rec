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


def get_poi_stats(lat: float, lon: float, query: str, radius: int) -> Tuple[int, Optional[float], List[str]]:
    """
    获取指定位置周边POI的统计信息
    
    Returns:
        (数量, 最近距离, POI名称列表)
    """
    pois = get_all_pois_in_radius(lat, lon, query, radius)
    
    count = len(pois)
    min_distance = None
    poi_names = []
    
    for poi in pois:
        # 获取名称
        name = poi.get("name", "")
        if name:
            poi_names.append(name)
        
        # 获取距离
        distance = poi.get("_distance")
        if distance is None:
            detail_info = poi.get("detail_info", {})
            distance = detail_info.get("distance")
        
        if distance is not None:
            try:
                dist_val = float(distance)
                if min_distance is None or dist_val < min_distance:
                    min_distance = dist_val
            except:
                pass
    
    return count, min_distance, poi_names[:5]  # 只返回前5个名称


@dataclass
class SiteMetrics:
    """地块POI指标"""
    index: int
    name: str
    lat: float
    lon: float
    
    # 交通指标
    subway_count: int = 0
    subway_min_dist: Optional[float] = None
    subway_names: List[str] = field(default_factory=list)
    
    bus_count: int = 0
    bus_min_dist: Optional[float] = None
    
    parking_count: int = 0
    parking_min_dist: Optional[float] = None
    
    train_count: int = 0
    train_min_dist: Optional[float] = None
    train_names: List[str] = field(default_factory=list)
    
    highway_count: int = 0
    highway_min_dist: Optional[float] = None
    
    # 生活配套
    bank_count: int = 0
    bank_min_dist: Optional[float] = None
    
    supermarket_count: int = 0
    supermarket_min_dist: Optional[float] = None
    
    restaurant_count: int = 0
    restaurant_min_dist: Optional[float] = None
    
    hospital_count: int = 0
    hospital_min_dist: Optional[float] = None
    hospital_names: List[str] = field(default_factory=list)
    
    pharmacy_count: int = 0
    pharmacy_min_dist: Optional[float] = None
    
    # 教育资源
    university_count: int = 0
    university_min_dist: Optional[float] = None
    university_names: List[str] = field(default_factory=list)
    
    school_count: int = 0
    school_min_dist: Optional[float] = None
    
    # 产业配套
    logistics_count: int = 0
    logistics_min_dist: Optional[float] = None
    
    industrial_park_count: int = 0
    industrial_park_min_dist: Optional[float] = None
    
    # 商业环境
    office_count: int = 0
    office_min_dist: Optional[float] = None
    
    hotel_count: int = 0
    hotel_min_dist: Optional[float] = None
    
    # 综合评分
    traffic_score: float = 0.0
    life_score: float = 0.0
    education_score: float = 0.0
    industry_score: float = 0.0
    business_score: float = 0.0
    total_score: float = 0.0


def process_single_site(idx: int, row: pd.Series) -> SiteMetrics:
    """处理单个地块的POI数据"""
    lat = float(row["lat"])
    lon = float(row["lon"])
    name = str(row.get("宗地坐落", f"地块{idx}"))[:40]
    
    print(f"\n[{idx+1}] {name}")
    print(f"    坐标: ({lat:.6f}, {lon:.6f})")
    
    metrics = SiteMetrics(index=idx, name=name, lat=lat, lon=lon)
    
    # === 交通设施 ===
    print("    [交通]", end=" ")
    
    # 地铁站
    count, dist, names = get_poi_stats(lat, lon, "地铁站", 1500)
    metrics.subway_count = count
    metrics.subway_min_dist = dist
    metrics.subway_names = names
    print(f"地铁:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    # 公交站
    count, dist, _ = get_poi_stats(lat, lon, "公交站", 500)
    metrics.bus_count = count
    metrics.bus_min_dist = dist
    print(f"公交:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    # 停车场
    count, dist, _ = get_poi_stats(lat, lon, "停车场", 1000)
    metrics.parking_count = count
    metrics.parking_min_dist = dist
    print(f"停车:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    # 火车站
    count, dist, names = get_poi_stats(lat, lon, "火车站", 5000)
    metrics.train_count = count
    metrics.train_min_dist = dist
    metrics.train_names = names
    print(f"火车:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    # 高速出入口
    count, dist, _ = get_poi_stats(lat, lon, "高速公路出入口", 5000)
    metrics.highway_count = count
    metrics.highway_min_dist = dist
    print(f"高速:{count}")
    time.sleep(API_RATE_LIMIT)
    
    # === 生活配套 ===
    print("    [生活]", end=" ")
    
    count, dist, _ = get_poi_stats(lat, lon, "银行", 1000)
    metrics.bank_count = count
    metrics.bank_min_dist = dist
    print(f"银行:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    count, dist, _ = get_poi_stats(lat, lon, "超市", 1000)
    metrics.supermarket_count = count
    metrics.supermarket_min_dist = dist
    print(f"超市:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    count, dist, _ = get_poi_stats(lat, lon, "餐厅", 500)
    metrics.restaurant_count = count
    metrics.restaurant_min_dist = dist
    print(f"餐饮:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    count, dist, names = get_poi_stats(lat, lon, "医院", 3000)
    metrics.hospital_count = count
    metrics.hospital_min_dist = dist
    metrics.hospital_names = names
    print(f"医院:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    count, dist, _ = get_poi_stats(lat, lon, "药店", 1000)
    metrics.pharmacy_count = count
    metrics.pharmacy_min_dist = dist
    print(f"药店:{count}")
    time.sleep(API_RATE_LIMIT)
    
    # === 教育资源 ===
    print("    [教育]", end=" ")
    
    count, dist, names = get_poi_stats(lat, lon, "大学", 5000)
    metrics.university_count = count
    metrics.university_min_dist = dist
    metrics.university_names = names
    print(f"高校:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    count, dist, _ = get_poi_stats(lat, lon, "学校", 2000)
    metrics.school_count = count
    metrics.school_min_dist = dist
    print(f"中小学:{count}")
    time.sleep(API_RATE_LIMIT)
    
    # === 产业配套 ===
    print("    [产业]", end=" ")
    
    count, dist, _ = get_poi_stats(lat, lon, "物流园区", 5000)
    metrics.logistics_count = count
    metrics.logistics_min_dist = dist
    print(f"物流:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    count, dist, _ = get_poi_stats(lat, lon, "工业园区", 5000)
    metrics.industrial_park_count = count
    metrics.industrial_park_min_dist = dist
    print(f"工业园:{count}")
    time.sleep(API_RATE_LIMIT)
    
    # === 商业环境 ===
    print("    [商业]", end=" ")
    
    count, dist, _ = get_poi_stats(lat, lon, "写字楼", 2000)
    metrics.office_count = count
    metrics.office_min_dist = dist
    print(f"写字楼:{count}", end=" ")
    time.sleep(API_RATE_LIMIT)
    
    count, dist, _ = get_poi_stats(lat, lon, "酒店", 2000)
    metrics.hotel_count = count
    metrics.hotel_min_dist = dist
    print(f"酒店:{count}")
    time.sleep(API_RATE_LIMIT)
    
    # 计算评分
    calculate_scores(metrics)
    print(f"    [评分] 交通:{metrics.traffic_score:.1f} 生活:{metrics.life_score:.1f} "
          f"教育:{metrics.education_score:.1f} 产业:{metrics.industry_score:.1f} "
          f"商业:{metrics.business_score:.1f} => 总分:{metrics.total_score:.1f}")
    
    return metrics


def calculate_scores(metrics: SiteMetrics):
    """
    计算各维度评分（0-10分）
    
    评分逻辑：
    - 基于POI数量和距离综合评估
    - 距离越近、数量越多，分数越高
    """
    
    # === 交通便利性评分 ===
    traffic = 0.0
    
    # 地铁（权重最高）
    if metrics.subway_count > 0:
        traffic += 3.0
        if metrics.subway_min_dist and metrics.subway_min_dist < 500:
            traffic += 1.5
        elif metrics.subway_min_dist and metrics.subway_min_dist < 1000:
            traffic += 0.8
    
    # 公交
    if metrics.bus_count > 0:
        traffic += 1.5
        if metrics.bus_count >= 3:
            traffic += 0.5
    
    # 停车场
    if metrics.parking_count > 0:
        traffic += 0.8
        if metrics.parking_count >= 5:
            traffic += 0.3
    
    # 火车站
    if metrics.train_count > 0:
        traffic += 1.0
        if metrics.train_min_dist and metrics.train_min_dist < 2000:
            traffic += 0.5
    
    # 高速出入口
    if metrics.highway_count > 0:
        traffic += 1.2
        if metrics.highway_min_dist and metrics.highway_min_dist < 3000:
            traffic += 0.5
    
    metrics.traffic_score = min(traffic, 10.0)
    
    # === 生活配套评分 ===
    life = 0.0
    
    # 银行
    if metrics.bank_count > 0:
        life += 1.0
    
    # 超市
    if metrics.supermarket_count > 0:
        life += 1.5
        if metrics.supermarket_count >= 3:
            life += 0.5
    
    # 餐饮
    if metrics.restaurant_count > 0:
        life += 2.0
        if metrics.restaurant_count >= 5:
            life += 1.0
    
    # 医院
    if metrics.hospital_count > 0:
        life += 2.0
        if metrics.hospital_min_dist and metrics.hospital_min_dist < 1500:
            life += 0.5
    
    # 药店
    if metrics.pharmacy_count > 0:
        life += 0.5
    
    metrics.life_score = min(life, 10.0)
    
    # === 教育资源评分 ===
    education = 0.0
    
    # 高校
    if metrics.university_count > 0:
        education += 4.0
        if metrics.university_count >= 2:
            education += 2.0
    
    # 中小学
    if metrics.school_count > 0:
        education += 2.0
        if metrics.school_count >= 3:
            education += 1.0
    
    metrics.education_score = min(education, 10.0)
    
    # === 产业配套评分 ===
    industry = 0.0
    
    # 物流园
    if metrics.logistics_count > 0:
        industry += 3.0
        if metrics.logistics_min_dist and metrics.logistics_min_dist < 3000:
            industry += 1.0
    
    # 工业园
    if metrics.industrial_park_count > 0:
        industry += 3.0
        if metrics.industrial_park_count >= 2:
            industry += 1.0
    
    metrics.industry_score = min(industry, 10.0)
    
    # === 商业环境评分 ===
    business = 0.0
    
    # 写字楼
    if metrics.office_count > 0:
        business += 3.0
        if metrics.office_count >= 5:
            business += 1.0
    
    # 酒店
    if metrics.hotel_count > 0:
        business += 2.5
        if metrics.hotel_count >= 3:
            business += 0.5
    
    metrics.business_score = min(business, 10.0)
    
    # === 综合评分 ===
    # 加权平均，交通和生活权重较高
    weights = {
        "traffic": 0.30,
        "life": 0.25,
        "education": 0.15,
        "industry": 0.15,
        "business": 0.15,
    }
    
    metrics.total_score = (
        metrics.traffic_score * weights["traffic"] +
        metrics.life_score * weights["life"] +
        metrics.education_score * weights["education"] +
        metrics.industry_score * weights["industry"] +
        metrics.business_score * weights["business"]
    ) * 10 / sum(weights.values())  # 归一化到0-10


def metrics_to_dict(metrics: SiteMetrics) -> dict:
    """将指标转换为字典格式"""
    return {
        # 交通
        "交通_地铁数量(1.5km)": metrics.subway_count,
        "交通_地铁最近距离(m)": metrics.subway_min_dist,
        "交通_地铁站名": "|".join(metrics.subway_names) if metrics.subway_names else "",
        "交通_公交数量(0.5km)": metrics.bus_count,
        "交通_公交最近距离(m)": metrics.bus_min_dist,
        "交通_停车数量(1km)": metrics.parking_count,
        "交通_停车最近距离(m)": metrics.parking_min_dist,
        "交通_火车数量(5km)": metrics.train_count,
        "交通_火车最近距离(m)": metrics.train_min_dist,
        "交通_火车站名": "|".join(metrics.train_names) if metrics.train_names else "",
        "交通_高速出入口数量(5km)": metrics.highway_count,
        "交通_高速最近距离(m)": metrics.highway_min_dist,
        "交通_便利评分(0-10)": round(metrics.traffic_score, 2),
        
        # 生活
        "生活_银行数量(1km)": metrics.bank_count,
        "生活_银行最近距离(m)": metrics.bank_min_dist,
        "生活_超市数量(1km)": metrics.supermarket_count,
        "生活_超市最近距离(m)": metrics.supermarket_min_dist,
        "生活_餐饮数量(0.5km)": metrics.restaurant_count,
        "生活_餐饮最近距离(m)": metrics.restaurant_min_dist,
        "生活_医院数量(3km)": metrics.hospital_count,
        "生活_医院最近距离(m)": metrics.hospital_min_dist,
        "生活_医院名称": "|".join(metrics.hospital_names) if metrics.hospital_names else "",
        "生活_药店数量(1km)": metrics.pharmacy_count,
        "生活_药店最近距离(m)": metrics.pharmacy_min_dist,
        "生活_便利评分(0-10)": round(metrics.life_score, 2),
        
        # 教育
        "教育_高校数量(5km)": metrics.university_count,
        "教育_高校最近距离(m)": metrics.university_min_dist,
        "教育_高校名称": "|".join(metrics.university_names) if metrics.university_names else "",
        "教育_中小学数量(2km)": metrics.school_count,
        "教育_中小学最近距离(m)": metrics.school_min_dist,
        "教育_资源评分(0-10)": round(metrics.education_score, 2),
        
        # 产业
        "产业_物流园数量(5km)": metrics.logistics_count,
        "产业_物流最近距离(m)": metrics.logistics_min_dist,
        "产业_工业园数量(5km)": metrics.industrial_park_count,
        "产业_工业园最近距离(m)": metrics.industrial_park_min_dist,
        "产业_配套评分(0-10)": round(metrics.industry_score, 2),
        
        # 商业
        "商业_写字楼数量(2km)": metrics.office_count,
        "商业_写字楼最近距离(m)": metrics.office_min_dist,
        "商业_酒店数量(2km)": metrics.hotel_count,
        "商业_酒店最近距离(m)": metrics.hotel_min_dist,
        "商业_环境评分(0-10)": round(metrics.business_score, 2),
        
        # 综合
        "综合_选址评分(0-10)": round(metrics.total_score, 2),
    }


def generate_context(row: pd.Series, metrics: SiteMetrics) -> str:
    """生成用于嵌入模型的context文本"""
    parts = []
    
    # 基本信息
    name = row.get("宗地坐落", "")
    usage = row.get("土地用途", "")
    area = row.get("宗地面积(平方米)", 0)
    price = row.get("挂牌起始价(万元)", 0)
    
    parts.append(f"该地块位于{name}，用途为{usage}，面积{area:.0f}平方米，起始价{price:.0f}万元。")
    
    # 交通情况
    traffic_parts = []
    if metrics.subway_count > 0:
        traffic_parts.append(f"1.5公里内有{metrics.subway_count}个地铁站")
        if metrics.subway_names:
            traffic_parts[-1] += f"（{metrics.subway_names[0]}等）"
        if metrics.subway_min_dist:
            traffic_parts[-1] += f"，最近{metrics.subway_min_dist:.0f}米"
    
    if metrics.bus_count > 0:
        traffic_parts.append(f"500米内有{metrics.bus_count}个公交站")
    
    if metrics.train_count > 0:
        traffic_parts.append(f"5公里内有{metrics.train_count}个火车站")
        if metrics.train_names:
            traffic_parts[-1] += f"（{metrics.train_names[0]}）"
    
    if metrics.highway_count > 0:
        traffic_parts.append(f"5公里内有{metrics.highway_count}个高速出入口")
    
    if traffic_parts:
        parts.append("交通方面：" + "；".join(traffic_parts) + "。")
    
    parts.append(f"交通便利评分{metrics.traffic_score:.1f}分。")
    
    # 生活配套
    life_parts = []
    if metrics.restaurant_count > 0:
        life_parts.append(f"500米内有{metrics.restaurant_count}家餐饮")
    if metrics.supermarket_count > 0:
        life_parts.append(f"1公里内有{metrics.supermarket_count}家超市")
    if metrics.hospital_count > 0:
        life_parts.append(f"3公里内有{metrics.hospital_count}家医院")
        if metrics.hospital_names:
            life_parts[-1] += f"（{metrics.hospital_names[0]}等）"
    
    if life_parts:
        parts.append("生活配套：" + "；".join(life_parts) + "。")
    
    # 教育资源
    if metrics.university_count > 0:
        edu_text = f"5公里内有{metrics.university_count}所高校"
        if metrics.university_names:
            edu_text += f"（{metrics.university_names[0]}等）"
        parts.append(edu_text + "。")
    
    # 产业配套
    if metrics.logistics_count > 0 or metrics.industrial_park_count > 0:
        ind_parts = []
        if metrics.logistics_count > 0:
            ind_parts.append(f"{metrics.logistics_count}个物流园区")
        if metrics.industrial_park_count > 0:
            ind_parts.append(f"{metrics.industrial_park_count}个工业园区")
        parts.append(f"周边5公里内有{'、'.join(ind_parts)}。")
    
    # 价格
    if area > 0 and price > 0:
        unit_price = price / area
        parts.append(f"单位面积价格约{unit_price:.4f}万元/平方米。")
    
    return "".join(parts)


def main():
    """主函数"""
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    
    input_path = os.path.join(data_dir, "land_transactions_with_coordinates_metrics.csv")
    output_path = os.path.join(data_dir, "land_transactions_with_poi_v2.csv")
    
    print("=" * 70)
    print("百度地图POI数据获取工具 v2")
    print("=" * 70)
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 读取数据
    df = pd.read_csv(input_path)
    print(f"\n共 {len(df)} 个地块待处理")
    
    # 处理每个地块
    all_metrics = []
    for idx, row in df.iterrows():
        try:
            metrics = process_single_site(idx, row)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"    [错误] {e}")
            # 创建空指标
            metrics = SiteMetrics(
                index=idx, 
                name=str(row.get("宗地坐落", f"地块{idx}")),
                lat=float(row["lat"]),
                lon=float(row["lon"])
            )
            all_metrics.append(metrics)
    
    # 构建结果DataFrame
    print("\n" + "=" * 70)
    print("生成输出数据...")
    
    # 保留原始基础列
    base_cols = ["宗地坐落", "土地用途", "宗地面积(平方米)", "挂牌起始价(万元)", "lon", "lat"]
    result_df = df[base_cols].copy()
    
    # 添加POI指标
    poi_data = [metrics_to_dict(m) for m in all_metrics]
    poi_df = pd.DataFrame(poi_data)
    result_df = pd.concat([result_df, poi_df], axis=1)
    
    # 生成新的context
    new_contexts = []
    for idx, (_, row) in enumerate(df.iterrows()):
        ctx = generate_context(row, all_metrics[idx])
        new_contexts.append(ctx)
    result_df["context"] = new_contexts
    
    # 计算价格
    result_df["价格_万元/㎡"] = result_df["挂牌起始价(万元)"] / result_df["宗地面积(平方米)"]
    
    # 保存
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 数据已保存: {output_path}")
    
    # 统计摘要
    print("\n" + "=" * 70)
    print("POI统计摘要")
    print("=" * 70)
    
    summary_cols = [
        ("交通_地铁数量(1.5km)", "地铁站"),
        ("交通_公交数量(0.5km)", "公交站"),
        ("交通_火车数量(5km)", "火车站"),
        ("生活_餐饮数量(0.5km)", "餐饮"),
        ("生活_医院数量(3km)", "医院"),
        ("教育_高校数量(5km)", "高校"),
    ]
    
    for col, name in summary_cols:
        if col in poi_df.columns:
            mean_val = poi_df[col].mean()
            max_val = poi_df[col].max()
            has_count = (poi_df[col] > 0).sum()
            print(f"{name}: 平均{mean_val:.1f}个, 最大{max_val}个, {has_count}/{len(poi_df)}个地块有覆盖")
    
    print(f"\n综合选址评分: 平均{poi_df['综合_选址评分(0-10)'].mean():.2f}, "
          f"最高{poi_df['综合_选址评分(0-10)'].max():.2f}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
