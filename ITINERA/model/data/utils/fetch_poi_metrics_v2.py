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
    ("高校", "141201", 5000, "教育"),        # 高等院校
    ("中小学", "141203", 2000, "教育"),      # 中学
    # 产业
    ("物流园", "170200", 5000, "产业"),      # 物流速递
    ("工业园", "170100", 5000, "产业"),      # 工厂
    # 商业
    ("写字楼", "120201", 2000, "商业"),      # 写字楼
    ("酒店", "100000", 2000, "商业"),        # 住宿服务
]


def search_around_poi(lon: float, lat: float, poi_type: str, radius: int,
                      page_num: int = 1, page_size: int = 25) -> dict:
    """
    高德地图周边搜索API
    
    Args:
        lon: 经度 (WGS84)
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
        "sortrule": "distance",
        "page_num": str(page_num),
        "page_size": str(page_size),
        "show_fields": "business",
    }
    
    for retry in range(MAX_RETRIES):
        try:
            response = requests.get(AMAP_API_URL, params=params, timeout=15)
            result = response.json()
            
            if result.get("status") == "1":
                return result
            else:
                infocode = result.get("infocode", "")
                if infocode == "10003":
                    print(f"[配额超限]", end=" ")
                    return {"status": "0", "pois": [], "count": "0"}
                elif infocode == "10004":
                    print(f"[限流]", end=" ", flush=True)
                    time.sleep(2)
                    continue
                else:
                    return {"status": "0", "pois": [], "count": "0"}
        except Exception as e:
            print(f"[异常:{e}]", end=" ")
            time.sleep(1)
    
    return {"status": "0", "pois": [], "count": "0"}


def extract_core_name(name: str, poi_type: str = "") -> str:
    """
    提取POI核心名称，去除出入口、分店等后缀
    
    适用于：
    - 火车站/地铁站：去除A口、B出口、进站口、出站口等
    - 停车场：去除出入口、分区等
    - 银行/超市：去除分行、分店等
    
    例如：
    - "广州火车站A口" -> "广州火车站"
    - "花城街站(C进站口)" -> "花城街站"
    - "融创茂商务楼6座停车场(出入口)" -> "融创茂商务楼停车场"
    - "中国银行花都支行ATM" -> "中国银行花都支行"
    """
    import re
    
    result = name.strip()
    
    # 通用：去除括号内的出入口信息
    result = re.sub(r'\([A-Za-z0-9东南西北进出站入口]+\)$', '', result)
    result = re.sub(r'（[A-Za-z0-9东南西北进出站入口]+）$', '', result)
    
    # 站点类（火车站、地铁站、公交站）
    station_patterns = [
        r'[-—]?[A-Fa-fA-F1-9][口号].*$',      # A口、B号口等
        r'[-—]?[东南西北][口出入站].*$',        # 东口、南出口等
        r'[-—]?[进出][站口].*$',               # 进站口、出站口
        r'[-—]?出入口.*$',
        r'[-—]?售票.*$',
        r'[-—]?候车.*$',
        r'[-—]?站台.*$',
    ]
    
    # 停车场类
    parking_patterns = [
        r'\(出入口\)$',
        r'（出入口）$',
        r'[-—]?出入口$',
        r'[-—]?[A-Z]区$',                      # A区、B区
        r'[-—]?[一二三四五六七八九十]+区$',      # 一区、二区
        r'[-—]?[0-9]+号?楼?$',                 # 去除楼号但保留主体
    ]
    
    # 银行/ATM类
    bank_patterns = [
        r'ATM$',
        r'自助.*$',
        r'离行式.*$',
    ]
    
    # 应用所有模式
    all_patterns = station_patterns + parking_patterns + bank_patterns
    for pattern in all_patterns:
        result = re.sub(pattern, '', result)
    
    # 对于停车场，进一步标准化（去除细微差异如"6栋"vs"6座"）
    if '停车场' in name:
        # 提取主体名称 + 停车场
        match = re.match(r'^(.+?)[0-9一二三四五六七八九十]+[栋座号层楼区]', result)
        if match:
            result = match.group(1) + '停车场'
    
    return result.strip()


def get_pois_with_details(lon: float, lat: float, poi_type: str, radius: int,
                          max_pages: int = 3, dedupe_stations: bool = False) -> Tuple[int, Optional[float], List[dict]]:
    """
    获取POI统计和详情
    
    Args:
        dedupe_stations: 是否对站点类POI去重（火车站、地铁站等）
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
            distance = poi.get("distance")
            poi_info = {
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "location": poi.get("location", ""),
                "distance": int(distance) if distance else None,
                "tel": poi.get("tel", ""),
                "type": poi.get("type", ""),
            }
            
            if poi_info["distance"] is not None and poi_info["distance"] <= radius:
                all_pois.append(poi_info)
            elif poi_info["distance"] is None:
                all_pois.append(poi_info)
        
        total = int(result.get("count", 0))
        if len(all_pois) >= total or len(pois) < 25:
            break
        
        page_num += 1
        time.sleep(API_RATE_LIMIT)
    
    # 对POI去重（火车站、地铁站、停车场、银行等）
    if dedupe_stations and all_pois:
        seen_names = {}  # {核心名称: poi_info}
        deduped_pois = []
        
        for poi in all_pois:
            core_name = extract_core_name(poi["name"], poi_type)
            if core_name not in seen_names:
                seen_names[core_name] = poi
                deduped_pois.append(poi)
            else:
                # 保留距离更近的
                existing = seen_names[core_name]
                if poi["distance"] is not None:
                    if existing["distance"] is None or poi["distance"] < existing["distance"]:
                        # 更新为更近的
                        idx = deduped_pois.index(existing)
                        deduped_pois[idx] = poi
                        seen_names[core_name] = poi
        
        all_pois = deduped_pois
    
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
    
    # 需要去重的POI类型（有多个出入口/分店/ATM等）
    DEDUPE_TYPES = {
        "火车站",    # 多个进出站口
        "地铁站",    # 多个出入口
        "公交站",    # 可能有多个站台
        "停车场",    # 多个出入口
        "银行",      # ATM和网点分开计数
        "高速出入口", # 入口和出口分开
    }
    
    current_category = None
    for poi_name, poi_type, radius, category in POI_TYPES:
        if category != current_category:
            if current_category is not None:
                print()
            print(f"    [{category}]", end=" ")
            current_category = category
        
        # 对需要去重的POI类型启用去重
        dedupe = poi_name in DEDUPE_TYPES
        count, min_dist, pois = get_pois_with_details(lon, lat, poi_type, radius, dedupe_stations=dedupe)
        
        metrics.poi_data[poi_name] = {
            "count": count,
            "min_distance": min_dist,
            "radius": radius,
            "pois": pois
        }
        
        print(f"{poi_name}:{count}", end=" ", flush=True)
        time.sleep(API_RATE_LIMIT)
    
    print()
    calculate_scores(metrics)
    print(f"    [评分] 交通:{metrics.traffic_score:.1f} 生活:{metrics.life_score:.1f} "
          f"教育:{metrics.education_score:.1f} 产业:{metrics.industry_score:.1f} "
          f"商业:{metrics.business_score:.1f}")
    
    return metrics


def calculate_scores(metrics: SiteMetrics):
    """计算各维度评分（0-10分）"""
    data = metrics.poi_data
    
    # 交通便利性评分
    traffic = 0.0
    if data.get("地铁站", {}).get("count", 0) > 0:
        traffic += 3.0
        dist = data["地铁站"].get("min_distance")
        if dist and dist < 500:
            traffic += 1.5
        elif dist and dist < 1000:
            traffic += 0.8
    if data.get("公交站", {}).get("count", 0) > 0:
        traffic += 1.5
        if data["公交站"]["count"] >= 3:
            traffic += 0.5
    if data.get("停车场", {}).get("count", 0) > 0:
        traffic += 0.8
        if data["停车场"]["count"] >= 5:
            traffic += 0.3
    if data.get("火车站", {}).get("count", 0) > 0:
        traffic += 1.0
        dist = data["火车站"].get("min_distance")
        if dist and dist < 2000:
            traffic += 0.5
    if data.get("高速出入口", {}).get("count", 0) > 0:
        traffic += 1.2
        dist = data["高速出入口"].get("min_distance")
        if dist and dist < 3000:
            traffic += 0.5
    metrics.traffic_score = min(traffic, 10.0)
    
    # 生活配套评分
    life = 0.0
    if data.get("银行", {}).get("count", 0) > 0:
        life += 1.0
    if data.get("超市", {}).get("count", 0) > 0:
        life += 1.5
        if data["超市"]["count"] >= 3:
            life += 0.5
    if data.get("餐饮", {}).get("count", 0) > 0:
        life += 2.0
        if data["餐饮"]["count"] >= 5:
            life += 1.0
    if data.get("医院", {}).get("count", 0) > 0:
        life += 2.0
        dist = data["医院"].get("min_distance")
        if dist and dist < 1500:
            life += 0.5
    if data.get("药店", {}).get("count", 0) > 0:
        life += 0.5
    metrics.life_score = min(life, 10.0)
    
    # 教育资源评分
    education = 0.0
    if data.get("高校", {}).get("count", 0) > 0:
        education += 4.0
        if data["高校"]["count"] >= 2:
            education += 2.0
    if data.get("中小学", {}).get("count", 0) > 0:
        education += 2.0
        if data["中小学"]["count"] >= 3:
            education += 1.0
    metrics.education_score = min(education, 10.0)
    
    # 产业配套评分
    industry = 0.0
    if data.get("物流园", {}).get("count", 0) > 0:
        industry += 3.0
        dist = data["物流园"].get("min_distance")
        if dist and dist < 3000:
            industry += 1.0
    if data.get("工业园", {}).get("count", 0) > 0:
        industry += 3.0
        if data["工业园"]["count"] >= 2:
            industry += 1.0
    metrics.industry_score = min(industry, 10.0)
    
    # 商业环境评分
    business = 0.0
    if data.get("写字楼", {}).get("count", 0) > 0:
        business += 3.0
        if data["写字楼"]["count"] >= 5:
            business += 1.0
    if data.get("酒店", {}).get("count", 0) > 0:
        business += 2.5
        if data["酒店"]["count"] >= 3:
            business += 0.5
    metrics.business_score = min(business, 10.0)


def metrics_to_csv_row(metrics: SiteMetrics) -> dict:
    """转换为CSV行数据"""
    data = metrics.poi_data
    
    def get_names(poi_type: str, limit: int = 3) -> str:
        pois = data.get(poi_type, {}).get("pois", [])
        names = [p["name"] for p in pois[:limit] if p.get("name")]
        return "|".join(names)
    
    return {
        "交通_地铁数量(1.5km)": data.get("地铁站", {}).get("count", 0),
        "交通_地铁最近距离(m)": data.get("地铁站", {}).get("min_distance"),
        "交通_地铁站名": get_names("地铁站"),
        "交通_公交数量(0.5km)": data.get("公交站", {}).get("count", 0),
        "交通_公交最近距离(m)": data.get("公交站", {}).get("min_distance"),
        "交通_停车数量(1km)": data.get("停车场", {}).get("count", 0),
        "交通_停车最近距离(m)": data.get("停车场", {}).get("min_distance"),
        "交通_火车数量(5km)": data.get("火车站", {}).get("count", 0),
        "交通_火车最近距离(m)": data.get("火车站", {}).get("min_distance"),
        "交通_火车站名": get_names("火车站"),
        "交通_高速数量(5km)": data.get("高速出入口", {}).get("count", 0),
        "交通_高速最近距离(m)": data.get("高速出入口", {}).get("min_distance"),
        "交通_便利评分(0-10)": round(metrics.traffic_score, 2),
        "生活_银行数量(1km)": data.get("银行", {}).get("count", 0),
        "生活_银行最近距离(m)": data.get("银行", {}).get("min_distance"),
        "生活_超市数量(1km)": data.get("超市", {}).get("count", 0),
        "生活_超市最近距离(m)": data.get("超市", {}).get("min_distance"),
        "生活_餐饮数量(0.5km)": data.get("餐饮", {}).get("count", 0),
        "生活_餐饮最近距离(m)": data.get("餐饮", {}).get("min_distance"),
        "生活_医院数量(3km)": data.get("医院", {}).get("count", 0),
        "生活_医院最近距离(m)": data.get("医院", {}).get("min_distance"),
        "生活_医院名称": get_names("医院"),
        "生活_药店数量(1km)": data.get("药店", {}).get("count", 0),
        "生活_药店最近距离(m)": data.get("药店", {}).get("min_distance"),
        "生活_便利评分(0-10)": round(metrics.life_score, 2),
        "教育_高校数量(5km)": data.get("高校", {}).get("count", 0),
        "教育_高校最近距离(m)": data.get("高校", {}).get("min_distance"),
        "教育_高校名称": get_names("高校"),
        "教育_中小学数量(2km)": data.get("中小学", {}).get("count", 0),
        "教育_中小学最近距离(m)": data.get("中小学", {}).get("min_distance"),
        "教育_资源评分(0-10)": round(metrics.education_score, 2),
        "产业_物流园数量(5km)": data.get("物流园", {}).get("count", 0),
        "产业_物流最近距离(m)": data.get("物流园", {}).get("min_distance"),
        "产业_工业园数量(5km)": data.get("工业园", {}).get("count", 0),
        "产业_工业园最近距离(m)": data.get("工业园", {}).get("min_distance"),
        "产业_配套评分(0-10)": round(metrics.industry_score, 2),
        "商业_写字楼数量(2km)": data.get("写字楼", {}).get("count", 0),
        "商业_写字楼最近距离(m)": data.get("写字楼", {}).get("min_distance"),
        "商业_酒店数量(2km)": data.get("酒店", {}).get("count", 0),
        "商业_酒店最近距离(m)": data.get("酒店", {}).get("min_distance"),
        "商业_环境评分(0-10)": round(metrics.business_score, 2),
    }


def generate_context(row: pd.Series, metrics: SiteMetrics) -> str:
    """生成用于嵌入模型的context文本"""
    data = metrics.poi_data
    parts = []
    
    name = row.get("宗地坐落", "")
    usage = row.get("土地用途", "")
    area = row.get("宗地面积(平方米)", 0)
    price = row.get("挂牌起始价(万元)", 0)
    
    parts.append(f"该地块位于{name}，用途为{usage}，面积{area:.0f}平方米，起始价{price:.0f}万元。")
    
    # 交通
    traffic_parts = []
    subway = data.get("地铁站", {})
    if subway.get("count", 0) > 0:
        s = f"1.5公里内有{subway['count']}个地铁站"
        if subway.get("min_distance"):
            s += f"，最近{subway['min_distance']}米"
        traffic_parts.append(s)
    bus = data.get("公交站", {})
    if bus.get("count", 0) > 0:
        traffic_parts.append(f"500米内有{bus['count']}个公交站")
    train = data.get("火车站", {})
    if train.get("count", 0) > 0:
        traffic_parts.append(f"5公里内有{train['count']}个火车站")
    highway = data.get("高速出入口", {})
    if highway.get("count", 0) > 0:
        traffic_parts.append(f"5公里内有{highway['count']}个高速出入口")
    if traffic_parts:
        parts.append("交通：" + "；".join(traffic_parts) + "。")
    parts.append(f"交通便利评分{metrics.traffic_score:.1f}分。")
    
    # 生活
    life_parts = []
    restaurant = data.get("餐饮", {})
    if restaurant.get("count", 0) > 0:
        life_parts.append(f"500米内有{restaurant['count']}家餐饮")
    supermarket = data.get("超市", {})
    if supermarket.get("count", 0) > 0:
        life_parts.append(f"1公里内有{supermarket['count']}家超市")
    hospital = data.get("医院", {})
    if hospital.get("count", 0) > 0:
        life_parts.append(f"3公里内有{hospital['count']}家医院")
    if life_parts:
        parts.append("生活配套：" + "；".join(life_parts) + "。")
    
    # 教育
    university = data.get("高校", {})
    if university.get("count", 0) > 0:
        parts.append(f"5公里内有{university['count']}所高校。")
    
    # 产业
    logistics = data.get("物流园", {})
    industrial = data.get("工业园", {})
    if logistics.get("count", 0) > 0 or industrial.get("count", 0) > 0:
        ind_parts = []
        if logistics.get("count", 0) > 0:
            ind_parts.append(f"{logistics['count']}个物流园区")
        if industrial.get("count", 0) > 0:
            ind_parts.append(f"{industrial['count']}个工业园区")
        parts.append(f"周边5公里内有{'、'.join(ind_parts)}。")
    
    # 价格
    if area > 0 and price > 0:
        unit_price = price / area
        parts.append(f"单位面积价格约{unit_price:.4f}万元/平方米。")
    
    return "".join(parts)


def save_poi_json(metrics: SiteMetrics, output_dir: str):
    """保存地块POI详情为JSON文件"""
    poi_json = {
        "site_index": metrics.index,
        "site_name": metrics.name,
        "location": {"lat": metrics.lat, "lon": metrics.lon},
        "scores": {
            "交通": metrics.traffic_score,
            "生活": metrics.life_score,
            "教育": metrics.education_score,
            "产业": metrics.industry_score,
            "商业": metrics.business_score,
        },
        "poi_details": {}
    }
    
    for poi_type, info in metrics.poi_data.items():
        poi_json["poi_details"][poi_type] = {
            "count": info.get("count", 0),
            "radius": info.get("radius", 0),
            "min_distance": info.get("min_distance"),
            "items": info.get("pois", [])
        }
    
    filename = f"site_{metrics.index:03d}_poi.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(poi_json, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    
    input_path = os.path.join(data_dir, "land_transactions_with_coordinates_metrics.csv")
    output_csv = os.path.join(data_dir, "land_transactions_with_poi_v2.csv")
    output_json_dir = os.path.join(data_dir, "poi_details")
    
    os.makedirs(output_json_dir, exist_ok=True)
    
    print("=" * 70)
    print("高德地图POI数据获取工具 v2")
    print("=" * 70)
    print(f"输入: {input_path}")
    print(f"CSV输出: {output_csv}")
    print(f"JSON输出: {output_json_dir}/")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    df = pd.read_csv(input_path)
    print(f"\n共 {len(df)} 个地块待处理")
    
    all_metrics = []
    for idx, row in df.iterrows():
        try:
            metrics = process_single_site(idx, row)
            all_metrics.append(metrics)
            save_poi_json(metrics, output_json_dir)
        except Exception as e:
            print(f"    [错误] {e}")
            metrics = SiteMetrics(index=idx, name=str(row.get("宗地坐落", f"地块{idx}")),
                                  lat=float(row["lat"]), lon=float(row["lon"]))
            all_metrics.append(metrics)
        
        if idx < len(df) - 1:
            print(f"    等待{SITE_INTERVAL}秒...")
            time.sleep(SITE_INTERVAL)
    
    print("\n" + "=" * 70)
    print("生成CSV数据...")
    
    base_cols = ["宗地坐落", "土地用途", "宗地面积(平方米)", "挂牌起始价(万元)", "lon", "lat"]
    result_df = df[base_cols].copy()
    
    poi_rows = [metrics_to_csv_row(m) for m in all_metrics]
    poi_df = pd.DataFrame(poi_rows)
    result_df = pd.concat([result_df, poi_df], axis=1)
    
    contexts = [generate_context(df.iloc[i], all_metrics[i]) for i in range(len(df))]
    result_df["context"] = contexts
    result_df["价格_万元/㎡"] = result_df["挂牌起始价(万元)"] / result_df["宗地面积(平方米)"]
    
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✓ CSV已保存: {output_csv}")
    print(f"✓ JSON已保存: {output_json_dir}/ ({len(all_metrics)}个文件)")
    
    print("\n" + "=" * 70)
    print("统计摘要")
    print("=" * 70)
    for col in ["交通_地铁数量(1.5km)", "交通_公交数量(0.5km)", "生活_餐饮数量(0.5km)",
                "教育_高校数量(5km)", "产业_工业园数量(5km)"]:
        if col in poi_df.columns:
            mean_val = poi_df[col].mean()
            has_count = (poi_df[col] > 0).sum()
            print(f"{col}: 平均{mean_val:.1f}, {has_count}/{len(poi_df)}个地块有覆盖")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
