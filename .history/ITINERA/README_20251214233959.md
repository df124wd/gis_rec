# 智能选址推荐系统 (ITINERA)

基于大语言模型(LLM)和多目标优化的智能地块选址推荐系统。

## 项目概述

本系统帮助用户根据自然语言描述的选址需求，从地块数据库中智能推荐最合适的地块。系统结合了：
- **语义检索**：基于文本嵌入的向量相似度匹配
- **LLM智能分析**：动态推导评价指标权重、区位评分、理想面积范围
- **多目标优化**：帕累托前沿算法平衡多个评价维度
- **地图可视化**：基于OpenLayers的交互式地图展示

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 (Web)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ OpenLayers  │  │   地图瓦片   │  │   推荐结果展示      │  │
│  │   地图组件   │  │  (高德影像)  │  │  (评分表格/分析)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    后端 (FastAPI)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  API路由    │  │  瓦片代理   │  │   地理编码代理      │  │
│  │ /api/...    │  │ /tiles/...  │  │   /api/geocode      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   核心算法 (SiteSelector)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ LLM权重推导 │  │  语义检索   │  │   多目标优化        │  │
│  │ (DeepSeek)  │  │ (Embedding) │  │  (Pareto Front)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据层                                  │
│  ┌─────────────────────────┐  ┌───────────────────────────┐ │
│  │  地块数据 (CSV)          │  │  文本嵌入向量 (NPY)       │ │
│  │  land_transactions_...  │  │  land_transactions_...    │ │
│  └─────────────────────────┘  └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
ITINERA/
├── server.py                 # FastAPI服务入口
├── requirements.txt          # Python依赖
├── model/
│   ├── site_selector.py      # 核心选址算法
│   ├── search.py             # 语义检索引擎
│   ├── spatial.py            # 空间处理模块
│   ├── multi_objective.py    # 多目标优化
│   ├── utils/
│   │   └── funcs.py          # 工具函数
│   └── data/
│       ├── land_transactions_with_coordinates_metrics.csv  # 地块数据
│       └── land_transactions_with_coordinates_metrics.npy  # 嵌入向量
├── web/
│   └── index.html            # 前端页面
└── static/
    └── lib/
        └── ol/               # OpenLayers库
```

## 环境配置

### 1. Python环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境变量

创建 `.env` 文件或设置环境变量：

```bash
# DeepSeek API配置（必需）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 高德地图API（可选，用于地理编码）
AMAP_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 启动服务

```bash
cd ITINERA
python server.py
```

服务启动后访问：http://localhost:8001

## 核心模块说明

### 1. SiteSelector (site_selector.py)

选址推荐的核心类，主要流程：

```python
selector = SiteSelector(
    user_reqs="花都区食品生产厂，交通便利，价格适中",
    proxy_call=llm_proxy,
    dataset_path="model/data/land_transactions_with_coordinates_metrics.csv"
)
result = selector.solve()
```

#### 处理流程

```
用户需求 (自然语言)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. parse_user_request()             │
│    LLM解析需求 → 结构化格式          │
│    输出: [{pos:"花都区", type:"区域"}]│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. derive_scoring_weights()         │
│    LLM推导指标权重                   │
│    输出: {traffic:0.25, price:0.35} │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. _derive_region_scores_with_llm() │
│    LLM推导区位评分                   │
│    输出: {花都区:10, 天河区:5, ...}  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. _derive_ideal_area_with_llm()    │
│    LLM推导理想面积范围               │
│    输出: {min:15000, max:40000}     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. get_candidate_sites()            │
│    语义检索候选地块                  │
│    输出: [(id, similarity_score)]   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. _multi_objective_selection()     │
│    多目标优化 (帕累托前沿)           │
│    输出: 最优地块列表                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 7. generate_recommendation()        │
│    生成推荐报告 + LLM分析优势/风险   │
│    输出: JSON结果                    │
└─────────────────────────────────────┘
```

### 2. 评价指标体系

系统使用4个核心评价指标：

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **交通便利性** | 地铁、公交、停车场等 | 直接使用数据集中的 `交通_便利评分(0-10)` |
| **性价比** | 单价越低越好 | `10 - 9*(单价-最低)/(最高-最低)` |
| **面积匹配度** | 与理想面积的匹配程度 | 基于LLM推导的理想范围计算 |
| **区位优势** | 行政区发展水平 | LLM根据用户需求动态评分 |

### 3. 大模型接口

系统使用DeepSeek API进行以下任务：

```python
# LLM调用封装
class LLMProxy:
    def chat(self, messages, model=None):
        # 调用DeepSeek Chat API
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "deepseek-chat", "messages": messages}
        )
        return response.json()["choices"][0]["message"]["content"]
```

LLM使用场景：
1. **需求解析**：将自然语言转为结构化格式
2. **权重推导**：根据需求分配指标权重
3. **区位评分**：根据业务类型评估各区域适合度
4. **面积推导**：推断理想的地块面积范围
5. **优势/风险分析**：为每个地块生成详细分析

### 4. 数据格式

#### 地块数据 (CSV)

```csv
宗地坐落,土地用途,宗地面积(平方米),挂牌起始价(万元),lon,lat,交通_便利评分(0-10),价格_万元/㎡,context
广州市花都区...,一类工业用地,29334.98,5292.0,113.16176,23.4787,4.78,0.180399,该地块位于...
```

主要字段：
- `宗地坐落`：地块位置描述
- `土地用途`：工业/商业/住宅等
- `宗地面积(平方米)`：地块面积
- `挂牌起始价(万元)`：总价
- `lon/lat`：经纬度坐标
- `交通_便利评分(0-10)`：交通便利性评分
- `交通_地铁数量(1.5km)`：1.5km内地铁站数量
- `交通_公交数量(0.5km)`：500m内公交站数量
- `价格_万元/㎡`：单价
- `context`：地块详细描述（用于语义检索）

#### 嵌入向量 (NPY)

```python
# 形状: (n_samples, embedding_dim)
# 例如: (39, 1536) 表示39个地块，每个1536维向量
embeddings = np.load("land_transactions_with_coordinates_metrics.npy")
```

## API接口

### POST /api/recommendations

生成选址推荐

**请求：**
```json
{
    "requirements": "花都区食品生产厂，交通便利，价格适中",
    "city": "guangzhou",
    "top_k": 10
}
```

**响应：**
```json
{
    "sites": {
        "1": {
            "id": "32",
            "name": "从化区到花都区花都大道西段1号（TZ）地块",
            "lat": 23.4294,
            "lon": 113.2372,
            "score": 8.99,
            "score_details": {
                "traffic": {"raw": 6.55, "normalized": 6.55, "desc": "6.55分"},
                "price": {"raw": 2955, "normalized": 9.84, "desc": "9.84分"},
                "area": {"raw": 23713, "normalized": 9.63, "desc": "9.63分"},
                "region": {"raw": 10.0, "normalized": 10.0, "desc": "10.00分"}
            },
            "advantages": ["交通便利（6.55分）", "性价比高（9.84分）"],
            "risks": ["建议实地考察确认"]
        }
    },
    "parsed_requirements": {
        "positive": ["花都区", "食品生产厂", "交通便利", "价格适中"],
        "negative": []
    },
    "weights": {
        "traffic": {"name": "交通便利性", "value": 0.25},
        "price": {"name": "价格成本", "value": 0.35},
        "area": {"name": "地块规模", "value": 0.25},
        "region": {"name": "区位优势", "value": 0.15}
    },
    "features": [...],
    "center": {"lon": 113.2, "lat": 23.4}
}
```

### GET /api/geocode

地理编码（地名转坐标）

**请求：**
```
GET /api/geocode?q=广州塔&city=广州
```

**响应：**
```json
{
    "lon": 113.324,
    "lat": 23.106,
    "address": "广州市海珠区阅江西路222号"
}
```

## 前端说明

### 技术栈
- **OpenLayers 7.x**：地图渲染
- **高德地图瓦片**：影像底图 + 标注层
- **原生JavaScript**：无框架依赖

### 主要功能
1. **需求输入**：自然语言描述选址需求
2. **需求解析展示**：显示拆解后的子需求和指标权重
3. **地图展示**：推荐地块标记在地图上
4. **评分表格**：每个地块的4维评分
5. **优势/风险分析**：LLM生成的详细分析
6. **地名搜索**：搜索地名并定位

## 扩展开发

### 添加新的评价指标

1. 在数据集中添加新列
2. 修改 `composite_score()` 方法
3. 更新 `derive_scoring_weights()` 的提示词
4. 更新前端展示

### 更换LLM服务

修改 `server.py` 中的 `LLMProxy` 类：

```python
class LLMProxy:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
    
    def chat(self, messages, model=None):
        # 适配其他LLM API
        pass
```

### 添加新城市数据

1. 准备CSV数据文件（参考现有格式）
2. 生成嵌入向量（使用 `SearchEngine` 自动生成）
3. 更新区位评分的行政区列表

## 常见问题

### Q: 地图显示空白？
A: 检查网络连接，高德地图瓦片需要联网加载。

### Q: LLM调用失败？
A: 检查 `DEEPSEEK_API_KEY` 环境变量是否正确设置。

### Q: 推荐结果不准确？
A: 可以调整 `min_site_candidate_num` 参数增加候选数量。

## 许可证

MIT License
