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


def process_single_site(idx: int, row: pd.Series) -> SiteMetrics:
    """处理单个地块的POI数据"""
    lat = float(row["lat"])
    lon = float(row["lon"])
    name = str(row.get("宗地坐落", f"地块{idx}"))[:40]
    
    print(f"\n[{idx+1}] {name}")
    print(f"    坐标: ({lat:.6f}, {lon:.6f})")
    
    metrics = SiteMetrics(index=idx, name=name, lat=lat, lon=lon)
    
    current_category = None
    for poi_type, query, radius, category in POI_TYPES:
        # 打印类别标题
        if category != current_category:
            if current_category is not None:
                print()  # 换行
            print(f"    [{category}]", end=" ")
            current_category = category
        
        # 获取POI
        count, min_dist, pois = get_pois_with_details(lat, lon, query, radius)
        
        # 存储数据
        metrics.poi_data[poi_type] = {
            "count": count,
            "min_distance": min_dist,
            "radius": radius,
            "pois": pois
        }
        
        print(f"{poi_type}:{count}", end=" ", flush=True)
        time.sleep(API_RATE_LIMIT)
    
    print()  # 换行
    
    # 计算评分
    calculate_scores(metrics)
    print(f"    [评分] 交通:{metrics.traffic_score:.1f} 生活:{metrics.life_score:.1f} "
          f"教育:{metrics.education_score:.1f} 产业:{metrics.industry_score:.1f} "
          f"商业:{metrics.business_score:.1f}")
    
    return metrics


def calculate_scores(metrics: SiteMetrics):
    """计算各维度评分（0-10分）"""
    data = metrics.poi_data
    
    # === 交通便利性评分 ===
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
    
    # === 生活配套评分 ===
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
    
    # === 教育资源评分 ===
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
    
    # === 产业配套评分 ===
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
    
    # === 商业环境评分 ===
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
        # 交通
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
        # 生活
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
        # 教育
        "教育_高校数量(5km)": data.get("高校", {}).get("count", 0),
        "教育_高校最近距离(m)": data.get("高校", {}).get("min_distance"),
        "教育_高校名称": get_names("高校"),
        "教育_中小学数量(2km)": data.get("中小学", {}).get("count", 0),
        "教育_中小学最近距离(m)": data.get("中小学", {}).get("min_distance"),
        "教育_资源评分(0-10)": round(metrics.education_score, 2),
        # 产业
        "产业_物流园数量(5km)": data.get("物流园", {}).get("count", 0),
        "产业_物流最近距离(m)": data.get("物流园", {}).get("min_distance"),
        "产业_工业园数量(5km)": data.get("工业园", {}).get("count", 0),
        "产业_工业园最近距离(m)": data.get("工业园", {}).get("min_distance"),
        "产业_配套评分(0-10)": round(metrics.industry_score, 2),
        # 商业
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
            s += f"，最近{subway['min_distance']:.0f}米"
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
    # 构建JSON数据
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
    
    # 按类别组织POI
    for poi_type, info in metrics.poi_data.items():
        poi_json["poi_details"][poi_type] = {
            "count": info.get("count", 0),
            "radius": info.get("radius", 0),
            "min_distance": info.get("min_distance"),
            "items": info.get("pois", [])
        }
    
    # 保存文件
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
    
    # 创建POI详情输出目录
    os.makedirs(output_json_dir, exist_ok=True)
    
    print("=" * 70)
    print("百度地图POI数据获取工具 v2")
    print("=" * 70)
    print(f"输入: {input_path}")
    print(f"CSV输出: {output_csv}")
    print(f"JSON输出: {output_json_dir}/")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API间隔: {API_RATE_LIMIT}秒, 地块间隔: {SITE_INTERVAL}秒")
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
            
            # 保存POI JSON
            save_poi_json(metrics, output_json_dir)
            
        except Exception as e:
            print(f"    [错误] {e}")
            metrics = SiteMetrics(
                index=idx,
                name=str(row.get("宗地坐落", f"地块{idx}")),
                lat=float(row["lat"]),
                lon=float(row["lon"])
            )
            all_metrics.append(metrics)
        
        # 地块间等待，避免限流
        if idx < len(df) - 1:
            print(f"    等待{SITE_INTERVAL}秒...")
            time.sleep(SITE_INTERVAL)
    
    # 构建CSV
    print("\n" + "=" * 70)
    print("生成CSV数据...")
    
    base_cols = ["宗地坐落", "土地用途", "宗地面积(平方米)", "挂牌起始价(万元)", "lon", "lat"]
    result_df = df[base_cols].copy()
    
    # 添加POI指标
    poi_rows = [metrics_to_csv_row(m) for m in all_metrics]
    poi_df = pd.DataFrame(poi_rows)
    result_df = pd.concat([result_df, poi_df], axis=1)
    
    # 生成context
    contexts = [generate_context(df.iloc[i], all_metrics[i]) for i in range(len(df))]
    result_df["context"] = contexts
    
    # 价格
    result_df["价格_万元/㎡"] = result_df["挂牌起始价(万元)"] / result_df["宗地面积(平方米)"]
    
    # 保存
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✓ CSV已保存: {output_csv}")
    print(f"✓ JSON已保存: {output_json_dir}/ ({len(all_metrics)}个文件)")
    
    # 统计
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
