"""
preprocess_data.py - Merge 5 index datasets into unified grid table.

Usage:
    python preprocess_data.py

Input: 5 CSV files from teacher (dwd_lvi/pel_cod/rri/sdi/uqi_zb_2022.csv)
Output: unified_grid.csv with all indicators merged, lon/lat decoded, ready for scoring.
"""
import pandas as pd
import numpy as np
import binascii
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__))
SRC_DIR = os.path.join(
    r'D:\gis_data\dk-data',
    '5\u5927\u6307\u6570\u6570\u636e\u96c6\uff08\u4e94\u534e\u3001\u56db\u4f1a\u3001\u53f0\u5c71\uff09',
    '5\u5927\u6307\u6570\u6570\u636e\u96c6\uff08\u4e94\u534e\u3001\u56db\u4f1a\u3001\u53f0\u5c71\uff09',
)

FILES = {
    'lvi':      ('dwd_lvi_zb_2022.csv',      'gbk'),
    'pel_cod':  ('dwd_pel_cod_zb_2022.csv',   'gbk'),
    'rri':      ('dwd_rri_zb_2022.csv',       'gbk'),
    'sdi':      ('dwd_sdi_zb_2022.csv',       'utf-8'),
    'uqi':      ('dwd_uqi_zb_2022.csv',       'utf-8'),
}

# Columns to always drop
DROP_COLS = {'create_time', 'update_time', 'valid_flag',
             '\u521b\u5efa\u65f6\u95f4', '\u66f4\u65b0\u65f6\u95f4', '\u6709\u6548\u6807\u8bc6'}

# Common key columns (shared across all 5 datasets)
KEY_COLS = ['geohash', 'code', '\u5e02\u7ea7\u4ee3\u7801', '\u5e02\u7ea7\u540d\u79f0',
            '\u53bf\u7ea7\u4ee3\u7801', '\u53bf\u7ea7\u540d\u79f0',
            '\u9547\u7ea7\u4ee3\u7801', '\u9547\u7ea7\u540d\u79f0', 'geom', '\u5e74\u4efd']


def decode_geom_wkb(geom_hex):
    """Decode PostGIS WKB hex to (lon, lat)."""
    if not isinstance(geom_hex, str) or len(geom_hex) < 20:
        return np.nan, np.nan
    try:
        from shapely import wkb as _wkb
        pt = _wkb.loads(binascii.unhexlify(geom_hex))
        return pt.x, pt.y
    except Exception:
        return np.nan, np.nan


def load_dataset(key):
    """Load one dataset, return DataFrame with indicator columns only (no keys/admin)."""
    fname, enc = FILES[key]
    fpath = os.path.join(SRC_DIR, fname)
    df = pd.read_csv(fpath, encoding=enc)
    # Drop admin/time columns
    drop = [c for c in df.columns if c in DROP_COLS]
    df.drop(columns=drop, inplace=True, errors='ignore')
    return df


def main():
    print('[1/5] Loading datasets...')
    datasets = {}
    for key in FILES:
        df = load_dataset(key)
        print(f'  {key}: {df.shape[0]} rows, {df.shape[1]} cols')
        datasets[key] = df

    # Verify row counts match
    row_counts = set(df.shape[0] for df in datasets.values())
    assert len(row_counts) == 1, f'Row count mismatch: {row_counts}'
    print(f'  All datasets: {list(row_counts)[0]} rows')

    # -----------------------------------------------------------------------
    # Merge on common key columns
    # -----------------------------------------------------------------------
    print('[2/5] Merging datasets on common keys...')
    # Start with lvi as base (contains all key columns)
    merged = datasets['lvi'].copy()

    # Track which indicator columns we've already seen (to handle duplicates)
    seen_cols = set(merged.columns)

    for key in ['pel_cod', 'rri', 'sdi', 'uqi']:
        df = datasets[key]
        # Separate key columns from indicator columns
        new_indicator_cols = [c for c in df.columns if c not in KEY_COLS and c not in DROP_COLS]

        # For duplicate indicator columns, rename with source prefix
        rename_map = {}
        for c in new_indicator_cols:
            if c in seen_cols:
                rename_map[c] = f'{key}__{c}'
            else:
                seen_cols.add(c)

        df_renamed = df.rename(columns=rename_map)

        # Drop key columns from right side (already in merged)
        merge_cols = [c for c in df_renamed.columns if c not in KEY_COLS or c == 'geohash']
        right = df_renamed[merge_cols]

        merged = merged.merge(right, on='geohash', how='left', suffixes=('', f'_dup_{key}'))
        print(f'  After merge {key}: {merged.shape[1]} cols')

    # Remove any _dup_ columns (exact duplicates from merge conflicts)
    dup_cols = [c for c in merged.columns if '_dup_' in c]
    merged.drop(columns=dup_cols, inplace=True)

    # -----------------------------------------------------------------------
    # Decode geom -> lon, lat
    # -----------------------------------------------------------------------
    print('[3/5] Decoding geom (WKB hex -> lon/lat)...')
    lons, lats = [], []
    geom_col = merged['geom']
    for i, val in enumerate(geom_col):
        lon, lat = decode_geom_wkb(val)
        lons.append(lon)
        lats.append(lat)
        if (i + 1) % 100000 == 0:
            print(f'  Decoded {i + 1}/{len(geom_col)}...')

    merged['lon'] = lons
    merged['lat'] = lats
    valid_coords = merged['lon'].notna().sum()
    print(f'  Valid coordinates: {valid_coords}/{len(merged)}')

    # Drop geom column (no longer needed)
    merged.drop(columns=['geom'], inplace=True)

    # -----------------------------------------------------------------------
    # Identify and organize indicator columns
    # -----------------------------------------------------------------------
    print('[4/5] Organizing indicators into 5 dimensions...')

    # All non-key, non-admin columns are indicators
    admin_cols = set(KEY_COLS) | {'lon', 'lat'}
    admin_cols.discard('geom')
    indicator_cols = [c for c in merged.columns if c not in admin_cols and c not in DROP_COLS]

    # Dimension grouping (by Chinese column name patterns)
    DIMENSIONS = {
        'eco_safety': [
            '\u751f\u6001\u4fdd\u62a4\u7ea2\u7ebf\u9762\u79ef',
            '\u996e\u7528\u6c34\u6c34\u6e90\u4fdd\u62a4\u533a\u9762\u79ef',
            '\u6cb3\u7f51\u5bc6\u5ea6',
            '\u5165\u6cb3\u6392\u6c61\u53e3\u6570\u91cf',
            '\u5730\u707e\u9690\u60a3\u70b9\u6570\u91cf',
            '\u6cb3\u6e56\u7ba1\u7406\u8303\u56f4\u9762\u79ef',
            '\u98ce\u66b4\u6f6e\u98ce\u9669\u7b49\u7ea7',
            '\u6797\u5730\u7834\u788e\u5ea6',
            '\u6e7f\u5730\u5f62\u6001\u6307\u6570',
        ],
        'pop_economy': [
            '\u5e38\u4f4f\u4eba\u53e3\u6570\u91cf',
            '\u52b3\u52a8\u5e74\u9f84\u4eba\u53e3\uff0818-60\u5c81\uff09\u6570\u91cf',
            '\u8001\u9f84\u4eba\u53e3\uff0861\u5c81\u4ee5\u4e0a\uff09\u6570\u91cf',
            '\u672c\u79d1\u53ca\u4ee5\u4e0a\u5b66\u5386\u4eba\u6570',
            '\u591c\u95f4\u706f\u5149\u4eae\u5ea6',
            '\u591c\u95f4\u706f\u5149\u4eae\u5ea6\u53d8\u5316',
            '\u7b2c\u4e8c\u3001\u4e09\u4ea7\u4e1a\u4f01\u4e1a\u6570\u91cf',
            '\u65b0\u5174\u4ea7\u4e1a\u4f01\u4e1a\u6570\u91cf',
            '\u56fd\u571f\u5f00\u53d1\u5f3a\u5ea6',
            '\u8015\u5730\u540e\u5907\u8d44\u6e90\u9762\u79ef',
            '\u4eba\u5747\u57ce\u4e61\u5efa\u8bbe\u7528\u5730\u9762\u79ef',
            '\u6c38\u4e45\u57fa\u672c\u519c\u7530\u4fdd\u62a4\u9762\u79ef',
        ],
        'resource_env': [
            '\u8015\u5730\u8fde\u7247\u5ea6',
            '\u9ad8\u6807\u51c6\u519c\u7530\u5360\u6bd4',
            '\u5de5\u4f5c\u4eba\u53e3\u6570\u91cf',
            '\u7cae\u98df\u4f5c\u7269\u9762\u79ef\u5360\u6bd4',
            '\u751f\u6001\u7cfb\u7edf\u670d\u52a1\u4ef7\u503c\u5f53\u91cf',
            '\u6797\u5730\u9762\u79ef\u5360\u6bd4',
            '\u65b0\u589e\u751f\u7269\u91cf',
            '\u6797\u5206\u4f18\u5316\u9762\u79ef',
            '\u8ddd\u4e3b\u8981\u9053\u8def\u8ddd\u79bb',
            '\u8ddd\u533b\u7597\u8bbe\u65bd\u4f4d\u7f6e\u8ddd\u79bb',
            '\u8ddd\u6559\u80b2\u8bbe\u65bd\u4f4d\u7f6e\u8ddd\u79bb',
            '\u4e09\u65e7\u6539\u9020\u6807\u56fe\u5efa\u5e93\u9762\u79ef',
            '\u5168\u57df\u571f\u6574\u5b50\u9879\u76ee\u9762\u79ef',
        ],
        'spatial_dev': [
            '\u57ce\u9547\u6751\u8303\u56f4\u5185\u5efa\u8bbe\u7528\u5730\u6bd4\u91cd',
            '\u4eba\u5747\u5efa\u8bbe\u7528\u5730\u9762\u79ef',
            '\u5e73\u5747\u5efa\u7b51\u5bc6\u5ea6',
            '\u4f01\u4e1a\u5bc6\u5ea6',
            '\u6218\u7565\u6027\u652f\u67f1\u4ea7\u4e1a\u4f01\u4e1a\u6570\u91cf',
            '\u9053\u8def\u5bc6\u5ea6',
            '\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u8986\u76d6\u5ea6',
            '\u5e02\u653f\u8bbe\u65bd\u5bc6\u5ea6',
            '\u5e73\u5747\u5761\u5ea6',
            '\u8ddd\u9ad8\u901f\u516c\u8def\u51fa\u5165\u53e3\u8ddd\u79bb',
            '\u57ce\u4e61\u5efa\u8bbe\u7528\u5730\u5f62\u6001\u6307\u6570',
        ],
        'urban_quality': [
            '\u4eba\u5747\u57ce\u9547\u5efa\u8bbe\u7528\u5730\u9762\u79ef',
            '\u6559\u80b2\u8bbe\u65bd\u8986\u76d6\u5ea6',
            '\u533b\u7597\u8bbe\u65bd\u8986\u76d6\u5ea6',
            '\u6587\u5316\u4f53\u80b2\u8bbe\u65bd\u8986\u76d6\u5ea6',
            '\u7eff\u8272\u5f00\u6563\u7a7a\u95f4\u8986\u76d6\u5ea6',
            '\u7eff\u5730\u7387',
            '\u5e73\u5747\u5bb9\u79ef\u7387',
            '\u6c34\u9762\u7387',
            '\u5c45\u4f4f\u7528\u5730\u5360\u6bd4',
            '\u4eba\u5747\u5c45\u4f4f\u7528\u5730\u9762\u79ef',
            '\u5b58\u91cf\u4f4e\u6548\u7528\u5730\u9762\u79ef',
            '\u57ce\u9547\u5f00\u53d1\u8fb9\u754c\u9762\u79ef',
        ],
    }

    # "Lower is better" indicators (inverted during normalization)
    INVERT_INDICATORS = {
        '\u751f\u6001\u4fdd\u62a4\u7ea2\u7ebf\u9762\u79ef',
        '\u996e\u7528\u6c34\u6c34\u6e90\u4fdd\u62a4\u533a\u9762\u79ef',
        '\u5165\u6cb3\u6392\u6c61\u53e3\u6570\u91cf',
        '\u5730\u707e\u9690\u60a3\u70b9\u6570\u91cf',
        '\u98ce\u66b4\u6f6e\u98ce\u9669\u7b49\u7ea7',
        '\u6797\u5730\u7834\u788e\u5ea6',
        '\u91c7\u77ff\u7528\u5730\u9762\u79ef',
        '\u8001\u9f84\u4eba\u53e3\uff0861\u5c81\u4ee5\u4e0a\uff09\u6570\u91cf',
        '\u8ddd\u4e3b\u8981\u9053\u8def\u8ddd\u79bb',
        '\u8ddd\u533b\u7597\u8bbe\u65bd\u4f4d\u7f6e\u8ddd\u79bb',
        '\u8ddd\u6559\u80b2\u8bbe\u65bd\u4f4d\u7f6e\u8ddd\u79bb',
        '\u5e73\u5747\u5761\u5ea6',
        '\u8ddd\u9ad8\u901f\u516c\u8def\u51fa\u5165\u53e3\u8ddd\u79bb',
    }

    # Verify all dimension indicators exist in merged
    dim_col_map = {}  # dimension -> list of actual column names
    for dim, cols in DIMENSIONS.items():
        found = []
        for c in cols:
            if c in merged.columns:
                found.append(c)
            else:
                # Check for prefixed versions (from duplicate handling)
                prefixed = [mc for mc in merged.columns if mc.endswith(f'__{c}')]
                if prefixed:
                    found.append(prefixed[0])
                else:
                    print(f'  WARNING: {dim} indicator "{c}" not found in merged data')
        dim_col_map[dim] = found

    # Print dimension summary
    total_indicators = 0
    for dim, cols in dim_col_map.items():
        print(f'  {dim}: {len(cols)} indicators')
        total_indicators += len(cols)
    print(f'  Total: {total_indicators} indicators')

    # -----------------------------------------------------------------------
    # Save dimension config as JSON for runtime use
    # -----------------------------------------------------------------------
    import json
    config = {
        'dimensions': {},
        'invert_indicators': list(INVERT_INDICATORS),
        'admin_cols': ['geohash', 'code', '\u5e02\u7ea7\u4ee3\u7801', '\u5e02\u7ea7\u540d\u79f0',
                       '\u53bf\u7ea7\u4ee3\u7801', '\u53bf\u7ea7\u540d\u79f0',
                       '\u9547\u7ea7\u4ee3\u7801', '\u9547\u7ea7\u540d\u79f0', '\u5e74\u4efd',
                       'lon', 'lat'],
    }
    for dim, cols in dim_col_map.items():
        config['dimensions'][dim] = cols

    config_path = os.path.join(DATA_DIR, 'dimension_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f'  Config saved to: {config_path}')

    # -----------------------------------------------------------------------
    # Handle NaN: fill with median
    # -----------------------------------------------------------------------
    print('[5/5] Filling NaN with column medians...')
    all_indicator_cols = []
    for cols in dim_col_map.values():
        all_indicator_cols.extend(cols)

    nan_counts = merged[all_indicator_cols].isna().sum()
    high_nan = nan_counts[nan_counts > len(merged) * 0.5]
    if len(high_nan) > 0:
        print('  Columns with >50% NaN:')
        for col, cnt in high_nan.items():
            print(f'    {col}: {cnt}/{len(merged)} ({cnt/len(merged)*100:.1f}%)')

    for col in all_indicator_cols:
        if col in merged.columns:
            median = merged[col].median()
            if pd.isna(median):
                median = 0
            merged[col] = merged[col].fillna(median)

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    out_path = os.path.join(DATA_DIR, 'unified_grid.csv')
    print(f'\nSaving unified grid to: {out_path}')
    print(f'  Shape: {merged.shape}')
    merged.to_csv(out_path, index=False, encoding='utf-8-sig')
    print('Done!')


if __name__ == '__main__':
    main()
