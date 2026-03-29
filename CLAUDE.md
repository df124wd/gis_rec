# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **GIS-REC**, an intelligent site selection recommendation system based on Large Language Models (LLM) and multi-objective optimization. The system recommends optimal land parcels for users based on natural language requirements, combining semantic search, LLM analysis, and Pareto optimization algorithms.

**Key Technologies:**
- **Backend**: Flask (Python) + DeepSeek/OpenAI LLM APIs
- **Frontend**: OpenLayers for map visualization, vanilla JavaScript
- **Data**: GeoSpatial CSV files with pre-computed embedding vectors (NPY)
- **Algorithms**: NSGA-II multi-objective optimization, cosine similarity search

## Quick Start Commands

```bash
# Install Python dependencies
cd ITINERA
pip install -r requirements.txt

# Run development server (default port 8001)
python server.py

# Run on custom port
PORT=8080 python server.py
```

**Access the application**: http://localhost:8001

## Configuration

The application reads configuration from `ITINERA/config/app_config.json` and environment variables:

**Required (for LLM features):**
- `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` - LLM API access
- `DEEPSEEK_BASE_URL` (optional, defaults to https://api.deepseek.com)

**Optional:**
- `AMAP_KEY` - Amap (高德地图) API for geocoding
- `TIANDITU_TK` - Tianditu (天地图) token for map tiles
- `EMBEDDING_PROVIDER` - Set to "local" to use local sentence-transformers models
- `LOCAL_EMBEDDING_MODEL` - Path to local embedding model (default: BAAI/bge-base-zh-v1.5)

## Architecture

### High-Level Flow

```
User Request (Natural Language)
    ↓
1. LLM parses requirements → structured format
    ↓
2. LLM derives evaluation weights (traffic/price/area/region)
    ↓
3. LLM derives district scores based on business type
    ↓
4. LLM derives ideal area range
    ↓
5. Semantic Search (embedding cosine similarity)
    ↓
6. Multi-Objective Optimization (NSGA-II / Pareto Front)
    ↓
7. LLM generates advantage/risk analysis for each site
    ↓
Recommendation Results (JSON + GeoJSON)
```

### Key Modules

| Module | File | Purpose |
|--------|------|---------|
| **Server** | `server.py` | Flask API, tile proxy, geocoding |
| **Site Selector** | `model/site_selector.py` | Core recommendation logic (2000+ lines) |
| **Search Engine** | `model/search.py` | Semantic search via embedding similarity |
| **Spatial Handler** | `model/spatial.py` | Spatial clustering, POI selection |
| **Multi-Objective** | `model/multi_objective.py` | NSGA-II optimization algorithm |
| **LLM Proxy** | `model/utils/proxy_call.py` | Unified DeepSeek/OpenAI interface |

### Data Format

**Land Parcel CSV** (`model/data/*.csv`):
- Required columns: `宗地坐落`, `lon`, `lat`, `土地用途`, `宗地面积(平方米)`, `挂牌起始价(万元)`
- Optional columns: `交通_便利评分(0-10)`, `交通_地铁数量(1.5km)`, `价格_万元/㎡`, `context`
- Embeddings are pre-computed and stored in corresponding `.npy` files

**Adding New Data:**
1. Prepare CSV with required columns
2. `SearchEngine` will auto-generate `.npy` embeddings on first run
3. Update `_derive_region_scores_with_llm()` district list if new city

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web UI |
| `/api/recommendations` | POST | Generate site recommendations |
| `/api/geocode` | GET | Geocode place name to coordinates |
| `/tiles/*` | GET | Tianditu tile proxy |
| `/api/poi_details/<site_index>` | GET | POI details for a site |
| `/examples/` | GET | OpenLayers examples |

## Common Development Tasks

### Adding a New Evaluation Metric

1. Add column to CSV data
2. Update `_derive_objectives_v2()` in `site_selector.py`
3. Update `derive_scoring_weights()` prompt
4. Update `composite_score()` calculation
5. Add frontend display in `web/index.html`

### Switching LLM Provider

Edit `model/utils/proxy_call.py`:
- Set `provider="deepseek"` or `provider="openai"`
- Configure respective API keys in environment

### Running Without LLM (Fallback)

The system has rule-based fallbacks:
- Default weights: traffic=0.30, price=0.25, area=0.20, region=0.25
- Default district scores: all districts = 7.0
- Default area range: 5000-50000 sq meters

### OpenLayers Development

- OpenLayers source is at `openlayers/src/ol/`
- Built examples are served from `/examples/`
- Use `http://localhost:8001/examples/` for testing

## Known Issues and Notes

1. **Embedding Dimension Mismatch**: If switching embedding providers, delete `.npy` files to regenerate
2. **DeepSeek Embedding**: DeepSeek doesn't support embedding API; system auto-switches to local models
3. **Concurrent LLM Calls**: Site analysis uses ThreadPoolExecutor with max_workers=5
4. **Removed Features**: SAFE inference, TSP routing, structured filters were removed as over-engineered

## File Structure Reference

```
gis_rec/
├── ITINERA/                    # Main application
│   ├── server.py               # Flask server
│   ├── requirements.txt        # Python dependencies
│   ├── model/
│   │   ├── site_selector.py    # Core algorithm
│   │   ├── search.py           # Semantic search
│   │   ├── multi_objective.py  # NSGA-II
│   │   ├── spatial.py          # Spatial utilities
│   │   ├── utils/
│   │   │   └── proxy_call.py   # LLM wrapper
│   │   └── data/
│   │       ├── *.csv           # Land parcel data
│   │       ├── *.npy           # Pre-computed embeddings
│   │       └── poi_details/    # Per-site POI JSON
│   ├── web/
│   │   └── index.html          # Frontend
│   └── config/
│       └── app_config.json     # Configuration
├── openlayers/                 # OpenLayers library
└── .venv/                      # Python virtual environment
```
