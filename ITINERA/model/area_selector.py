"""
area_selector.py - Area-based recommendation engine using 5-index grid data.

Replaces site_selector.py. Recommends suitable AREAS (clusters of geohash grids)
instead of individual land parcels.

Pipeline: Parse Requirements -> Derive Weights -> Score Grids -> DBSCAN Cluster
          -> Rank Areas -> LLM Analysis -> JSON Output
"""
import pandas as pd
import numpy as np
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor

from model.utils.proxy_call import OpenaiCall
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
GRID_CSV = os.path.join(DATA_DIR, 'unified_grid.csv')
DIM_CONFIG = os.path.join(DATA_DIR, 'dimension_config.json')

DIMENSION_NAMES = {
    'eco_safety': '\u751f\u6001\u5b89\u5168',
    'pop_economy': '\u4eba\u53e3\u7ecf\u6d4e',
    'resource_env': '\u8d44\u6e90\u73af\u5883',
    'spatial_dev': '\u7a7a\u95f4\u53d1\u5c55',
    'urban_quality': '\u57ce\u5e02\u8d28\u91cf',
}

DEFAULT_WEIGHTS = {
    'eco_safety': 0.20,
    'pop_economy': 0.20,
    'resource_env': 0.20,
    'spatial_dev': 0.20,
    'urban_quality': 0.20,
}


# ---------------------------------------------------------------------------
# AreaRecommender
# ---------------------------------------------------------------------------
class AreaRecommender:
    """Area-based recommendation engine using multi-index grid data."""

    def __init__(self):
        self._llm = None
        self._grid_df = None
        self._dim_config = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------
    def _ensure_llm(self):
        if self._llm is None:
            self._llm = OpenaiCall()
        return self._llm

    def _ensure_data(self):
        if self._grid_df is not None:
            return
        logger.info('[AreaSelector] Loading grid data...')
        self._grid_df = pd.read_csv(GRID_CSV, encoding='utf-8-sig')
        with open(DIM_CONFIG, 'r', encoding='utf-8') as f:
            self._dim_config = json.load(f)
        logger.info(f'[AreaSelector] Loaded {len(self._grid_df)} grids, '
                     f'{len(self._dim_config["dimensions"])} dimensions')

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _call_llm(self, system_prompt, user_prompt, temperature=0):
        llm = self._ensure_llm()
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        return llm.chat(messages, temperature=temperature)

    # ------------------------------------------------------------------
    # Step 1: Parse requirements with LLM
    # ------------------------------------------------------------------
    def parse_requirements(self, requirements_text):
        """Parse natural language requirements into structured format."""
        system_prompt = (
            "You are a site selection analyst. Parse the user's requirements "
            "into structured format. Output JSON ONLY, no markdown.\n"
            "Fields:\n"
            "- positive: list of positive requirements\n"
            "- negative: list of things to avoid\n"
            "- hard_constraints: list of must-have constraints\n"
            "- region: specific region if mentioned (e.g. '四会市'), else null\n"
            "- industry_type: type of use if mentioned (e.g. '工业', '商业', '居住'), else null\n"
        )
        user_prompt = f"Parse this site selection requirement: {requirements_text}"

        try:
            resp = self._call_llm(system_prompt, user_prompt, temperature=0)
            resp = resp.strip()
            if resp.startswith('```'):
                resp = resp.split('\n', 1)[-1].rsplit('```', 1)[0]
            parsed = json.loads(resp)
            parsed['original'] = requirements_text
            return parsed
        except Exception as e:
            logger.warning(f'[AreaSelector] Parse error: {e}')
            return {
                'positive': [requirements_text],
                'negative': [],
                'hard_constraints': [],
                'region': None,
                'industry_type': None,
                'original': requirements_text,
            }

    # ------------------------------------------------------------------
    # Step 2: Derive 5-dimension weights with LLM
    # ------------------------------------------------------------------
    def derive_dimension_weights(self, requirements_text, parsed=None):
        """Use LLM to derive weights for 5 evaluation dimensions."""
        system_prompt = (
            "You are a site selection expert. Based on the user's requirements, "
            "assign weights to 5 evaluation dimensions. Output JSON ONLY.\n\n"
            "Dimensions:\n"
            "- eco_safety (生态安全): ecological protection, geological safety, flood risk\n"
            "- pop_economy (人口经济): population, workforce, economic vitality\n"
            "- resource_env (资源环境): natural resources, farmland quality, ecology value\n"
            "- spatial_dev (空间发展): infrastructure, transportation, enterprise density\n"
            "- urban_quality (城市质量): public services, green space, living quality\n\n"
            "Output format: {\"eco_safety\": 0.xx, \"pop_economy\": 0.xx, "
            "\"resource_env\": 0.xx, \"spatial_dev\": 0.xx, \"urban_quality\": 0.xx}\n"
            "Weights MUST sum to 1.0"
        )
        user_prompt = f"Requirement: {requirements_text}"

        try:
            resp = self._call_llm(system_prompt, user_prompt, temperature=0)
            resp = resp.strip()
            if resp.startswith('```'):
                resp = resp.split('\n', 1)[-1].rsplit('```', 1)[0]
            weights = json.loads(resp)

            # Validate and normalize
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("Weights sum to zero")
            weights = {k: v / total for k, v in weights.items()}

            # Ensure all 5 dimensions present
            for dim in DIMENSION_NAMES:
                if dim not in weights:
                    weights[dim] = 0.1
            # Re-normalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            return weights
        except Exception as e:
            logger.warning(f'[AreaSelector] Weight error: {e}, using defaults')
            return dict(DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Step 3: Score grids using weighted dimensions
    # ------------------------------------------------------------------
    def score_grids(self, weights, region=None):
        """Score all grids using weighted 5-dimension evaluation."""
        self._ensure_data()
        df = self._grid_df

        # Filter by region if specified
        if region and region != 'all':
            county_col = '\u53bf\u7ea7\u540d\u79f0'
            if county_col in df.columns:
                df = df[df[county_col] == region].copy()
                logger.info(f'[AreaSelector] Filtered to {region}: {len(df)} grids')

        if len(df) == 0:
            raise ValueError(f"No grids found for region: {region}")

        dims = self._dim_config['dimensions']
        invert_cols = set(self._dim_config.get('invert_indicators', []))

        # Compute dimension scores
        dim_scores = {}
        for dim_key, cols in dims.items():
            existing_cols = [c for c in cols if c in df.columns]
            if not existing_cols:
                dim_scores[dim_key] = pd.Series(0.5, index=df.index)
                continue

            dim_data = df[existing_cols].copy()

            # Normalize each indicator to [0, 1]
            for col in existing_cols:
                col_min = dim_data[col].min()
                col_max = dim_data[col].max()
                if col_max > col_min:
                    dim_data[col] = (dim_data[col] - col_min) / (col_max - col_min)
                else:
                    dim_data[col] = 0.5

                # Invert "lower is better" indicators
                if col in invert_cols:
                    dim_data[col] = 1.0 - dim_data[col]

            dim_scores[dim_key] = dim_data.mean(axis=1)

        # Compute composite score
        composite = pd.Series(0.0, index=df.index)
        for dim_key, score_series in dim_scores.items():
            w = weights.get(dim_key, 0.2)
            composite += w * score_series

        # Scale to [1, 10]
        c_min, c_max = composite.min(), composite.max()
        if c_max > c_min:
            composite = 1 + 9 * (composite - c_min) / (c_max - c_min)
        else:
            composite = pd.Series(5.0, index=df.index)

        # Store results
        df = df.copy()
        df['composite_score'] = composite
        for dim_key, score_series in dim_scores.items():
            # Scale dimension scores to [1, 10] too
            s_min, s_max = score_series.min(), score_series.max()
            if s_max > s_min:
                df[f'dim_{dim_key}'] = 1 + 9 * (score_series - s_min) / (s_max - s_min)
            else:
                df[f'dim_{dim_key}'] = 5.0

        return df

    # ------------------------------------------------------------------
    # Step 4: Cluster high-scoring grids into areas (DBSCAN)
    # ------------------------------------------------------------------
    def cluster_areas(self, scored_df, top_pct=15, eps_km=0.8, min_grids=5):
        """Cluster top-scoring grids into recommended areas using DBSCAN."""
        n_top = max(int(len(scored_df) * top_pct / 100), 50)
        top = scored_df.nlargest(n_top, 'composite_score')

        logger.info(f'[AreaSelector] Top {top_pct}% = {len(top)} grids')

        # DBSCAN clustering on lon/lat
        coords = top[['lon', 'lat']].values
        # eps in degrees (approx): 0.8km / 111km per degree ~ 0.0072
        eps_deg = eps_km / 111.0
        clustering = DBSCAN(eps=eps_deg, min_samples=min_grids).fit(coords)
        top = top.copy()
        top['cluster'] = clustering.labels_

        # Filter out noise (-1)
        clustered = top[top['cluster'] >= 0]
        n_clusters = clustered['cluster'].nunique()
        logger.info(f'[AreaSelector] Found {n_clusters} areas '
                     f'({len(clustered)} grids clustered, '
                     f'{len(top) - len(clustered)} noise)')

        return clustered

    # ------------------------------------------------------------------
    # Step 5: Aggregate clusters into area objects
    # ------------------------------------------------------------------
    def aggregate_areas(self, clustered_df):
        """Compute area-level statistics from clustered grids."""
        areas = []
        township_col = '\u9547\u7ea7\u540d\u79f0'

        for cluster_id, group in clustered_df.groupby('cluster'):
            # Dominant township
            if township_col in group.columns:
                township = group[township_col].mode().iloc[0]
            else:
                township = f'Area-{cluster_id}'

            # Scores
            avg_score = group['composite_score'].mean()
            dim_scores = {}
            for dim_key in DIMENSION_NAMES:
                col = f'dim_{dim_key}'
                if col in group.columns:
                    dim_scores[dim_key] = round(group[col].mean(), 2)
                else:
                    dim_scores[dim_key] = 5.0

            # Center point
            center_lon = group['lon'].mean()
            center_lat = group['lat'].mean()

            # Convex hull for polygon boundary
            boundary = self._compute_boundary(group[['lon', 'lat']].values)

            # Area size (approximate: each geohash ~0.25 km2 at this resolution)
            grid_count = len(group)
            area_km2 = round(grid_count * 0.25, 2)

            # Find direction from township center
            direction = self._compute_direction(group, township_col)

            area_name = f'{township}{direction}适宜区域'

            areas.append({
                'name': area_name,
                'township': township,
                'score': round(avg_score, 2),
                'grid_count': grid_count,
                'area_km2': area_km2,
                'center_lon': round(center_lon, 6),
                'center_lat': round(center_lat, 6),
                'boundary': boundary,
                'dim_scores': dim_scores,
                'cluster_id': int(cluster_id),
            })

        # Sort by score descending
        areas.sort(key=lambda a: a['score'], reverse=True)
        return areas

    def _compute_boundary(self, coords):
        """Compute convex hull polygon from grid coordinates."""
        if len(coords) < 3:
            return None
        try:
            hull = ConvexHull(coords)
            hull_pts = coords[hull.vertices]
            # Close the polygon
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            return hull_pts.tolist()
        except Exception:
            # Fallback: bounding box
            lon_min, lat_min = coords.min(axis=0)
            lon_max, lat_max = coords.max(axis=0)
            return [
                [lon_min, lat_min], [lon_max, lat_min],
                [lon_max, lat_max], [lon_min, lat_max],
                [lon_min, lat_min],
            ]

    def _compute_direction(self, group, township_col):
        """Compute direction suffix based on offset from township center."""
        center_lon = group['lon'].mean()
        center_lat = group['lat'].mean()

        if township_col not in group.columns:
            return ''

        township = group[township_col].mode().iloc[0]
        # Use the first grid's township center as reference
        township_grids = self._grid_df[
            self._grid_df[township_col] == township
        ]
        if len(township_grids) == 0:
            return ''

        ref_lon = township_grids['lon'].mean()
        ref_lat = township_grids['lat'].mean()

        dlon = center_lon - ref_lon
        dlat = center_lat - ref_lat

        if abs(dlon) < 0.005 and abs(dlat) < 0.005:
            return '中部'

        dirs = []
        if dlat > 0.005:
            dirs.append('北')
        elif dlat < -0.005:
            dirs.append('南')
        if dlon > 0.005:
            dirs.append('东')
        elif dlon < -0.005:
            dirs.append('西')

        return ''.join(dirs) if dirs else ''

    # ------------------------------------------------------------------
    # Step 6: LLM area analysis (advantages / risks)
    # ------------------------------------------------------------------
    def generate_area_analysis(self, area, requirements_text=''):
        """Generate advantages and risks for a single area using LLM."""
        dim_desc = '\n'.join(
            f'- {DIMENSION_NAMES[k]}: {v}/10'
            for k, v in area['dim_scores'].items()
        )

        system_prompt = (
            "You are a professional site selection consultant. "
            "Based on the area's evaluation scores, generate concise analysis.\n"
            "Output JSON ONLY: {\"advantages\": [\"...\", ...], \"risks\": [\"...\", ...]}\n"
            "Each item should be 1-2 sentences, specific to the scores."
        )
        user_prompt = (
            f"Area: {area['name']}\n"
            f"Size: {area['area_km2']} km2 ({area['grid_count']} grid cells)\n"
            f"Composite score: {area['score']}/10\n"
            f"Dimension scores:\n{dim_desc}\n"
            f"User requirement: {requirements_text}"
        )

        try:
            resp = self._call_llm(system_prompt, user_prompt, temperature=0.3)
            resp = resp.strip()
            if resp.startswith('```'):
                resp = resp.split('\n', 1)[-1].rsplit('```', 1)[0]
            result = json.loads(resp)
            return result.get('advantages', []), result.get('risks', [])
        except Exception as e:
            logger.warning(f'[AreaSelector] Analysis error: {e}')
            # Fallback: generate from scores
            advs, risks = [], []
            for k, v in area['dim_scores'].items():
                if v >= 7:
                    advs.append(f'{DIMENSION_NAMES[k]}评分较高({v}/10)')
                elif v <= 4:
                    risks.append(f'{DIMENSION_NAMES[k]}评分偏低({v}/10)')
            return advs or ['综合评分良好'], risks or ['建议实地考察确认']

    # ------------------------------------------------------------------
    # Step 7: Build output JSON
    # ------------------------------------------------------------------
    def _build_output(self, areas, weights, parsed_reqs):
        """Build the final JSON response for the API."""
        sites = {}
        features = []

        for i, area in enumerate(areas, 1):
            idx = str(i)
            sites[idx] = {
                'id': str(area['cluster_id']),
                'name': area['name'],
                'score': area['score'],
                'site_index': i - 1,
                'lon': area['center_lon'],
                'lat': area['center_lat'],
                'center_lon': area['center_lon'],
                'center_lat': area['center_lat'],
                'area_km2': area['area_km2'],
                'grid_count': area['grid_count'],
                'township': area['township'],
                'advantages': area.get('advantages', []),
                'risks': area.get('risks', []),
                'score_details': {
                    k: {
                        'normalized': v,
                        'desc': f'{v:.1f}\u5206',
                    }
                    for k, v in area['dim_scores'].items()
                },
            }

            # GeoJSON feature with polygon
            if area['boundary']:
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [area['boundary']],
                    },
                    'properties': {
                        'index': i,
                        'id': str(area['cluster_id']),
                        'name': area['name'],
                        'score': area['score'],
                        'grid_count': area['grid_count'],
                    },
                }
                features.append(feature)

        # Center point (mean of top areas)
        if areas:
            center_lon = np.mean([a['center_lon'] for a in areas[:5]])
            center_lat = np.mean([a['center_lat'] for a in areas[:5]])
        else:
            center_lon, center_lat = 112.7, 23.2

        weights_output = {
            k: {'name': DIMENSION_NAMES[k], 'value': round(v, 2)}
            for k, v in weights.items()
        }

        return {
            'sites': sites,
            'features': features,
            'geojson': {'type': 'FeatureCollection', 'features': features},
            'center': {'lon': round(center_lon, 6), 'lat': round(center_lat, 6)},
            'weights': weights_output,
            'parsed_requirements': parsed_reqs,
            'recommendations': ' -> '.join(a['name'] for a in areas[:5]),
            'summary': f'共推荐 {len(areas)} 个适宜区域',
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def recommend(self, requirements, region='all', top_k=5):
        """
        Main recommendation pipeline.

        Args:
            requirements: Natural language requirements text
            region: Region filter ('all', '台山市', '四会市', '五华县')
            top_k: Number of areas to return

        Returns:
            Dict with sites, features, weights, etc.
        """
        self._ensure_data()
        logger.info(f'[AreaSelector] recommend(region={region}, top_k={top_k})')

        # Step 1: Parse requirements
        logger.info('[AreaSelector] Step 1: Parsing requirements...')
        parsed = self.parse_requirements(requirements)

        # Use parsed region if available
        if parsed.get('region') and region == 'all':
            region = parsed['region']

        # Step 2: Derive weights
        logger.info('[AreaSelector] Step 2: Deriving dimension weights...')
        weights = self.derive_dimension_weights(requirements, parsed)

        # Step 3: Score grids
        logger.info('[AreaSelector] Step 3: Scoring grids...')
        scored_df = self.score_grids(weights, region=region)

        # Step 4: Cluster into areas
        logger.info('[AreaSelector] Step 4: Clustering areas...')
        clustered = self.cluster_areas(scored_df)

        if len(clustered) == 0:
            logger.warning('[AreaSelector] No clusters found, relaxing params...')
            clustered = self.cluster_areas(scored_df, top_pct=25, eps_km=1.2, min_grids=3)

        # Step 5: Aggregate
        logger.info('[AreaSelector] Step 5: Aggregating areas...')
        areas = self.aggregate_areas(clustered)

        # Limit to top_k
        areas = areas[:top_k]

        if not areas:
            return {
                'error': 'No suitable areas found',
                'sites': {},
                'features': [],
                'weights': {k: {'name': DIMENSION_NAMES[k], 'value': v}
                            for k, v in weights.items()},
                'parsed_requirements': parsed,
            }

        # Step 6: LLM analysis for each area
        logger.info('[AreaSelector] Step 6: Generating area analysis...')
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self.generate_area_analysis, area, requirements
                ): i
                for i, area in enumerate(areas)
            }
            for future in futures:
                idx = futures[future]
                try:
                    advs, risks = future.result()
                    areas[idx]['advantages'] = advs
                    areas[idx]['risks'] = risks
                except Exception as e:
                    logger.warning(f'Analysis failed for area {idx}: {e}')
                    areas[idx]['advantages'] = ['综合评分良好']
                    areas[idx]['risks'] = ['建议实地考察确认']

        # Step 7: Build output
        logger.info('[AreaSelector] Step 7: Building output...')
        result = self._build_output(areas, weights, parsed)

        logger.info(f'[AreaSelector] Done! {len(areas)} areas recommended.')
        return result
