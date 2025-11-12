import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory

# Import SiteSelector and SimpleProxy
from flask import Flask, jsonify, request, send_from_directory
import os

from model.site_selector import SiteSelector
# 替换 SimpleProxy 为支持 base_url 的 OpenaiCall
from model.utils.proxy_call import OpenaiCall

app = Flask(__name__, static_folder='web', static_url_path='/static')

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('itinera')

# 配置加载
CONFIG = {}
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config', 'app_config.json'))
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            CONFIG = json.load(f) or {}
        logger.info('已加载配置: %s', CONFIG_PATH)
    except Exception as e:
        logger.error('加载配置失败: %s', e)

def _apply_env_from_config():
    keys = [
        'OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_PROXY_BASE',
        'OPENAI_API_KEY', 'DEEPSEEK_API_KEY',
        'OPENAI_CHAT_MODEL', 'OPENAI_EMBEDDING_MODEL'
    ]
    for k in keys:
        v = CONFIG.get(k)
        if isinstance(v, str):
            v = v.strip()
        if v:
            if k in os.environ:
                continue
            os.environ[k] = v
            if 'KEY' in k:
                logger.info('%s 已设置', k)
            else:
                logger.info('%s=%s', k, v)

_apply_env_from_config()

# Serve local OpenLayers ES modules and CSS from the downloaded repository
OL_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openlayers', 'src', 'ol'))

@app.route('/static/lib/ol/<path:filename>')
def serve_openlayers(filename):
    return send_from_directory(OL_SRC_DIR, filename)

@app.route('/api/config', methods=['GET'])
def get_config():
    data = {
        'tianditu_tk': (CONFIG.get('TIANDITU_TK') or os.environ.get('TIANDITU_TK') or ''),
        'amap_city_default': CONFIG.get('AMAP_CITY_DEFAULT') or ''
    }
    logger.info('配置查询: %s', data)
    return jsonify(data)

from flask import Response
import random

# ------------------- OpenLayers examples build serving -------------------
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openlayers', 'build', 'examples'))

@app.route('/examples/')
def serve_examples_index():
    return send_from_directory(EXAMPLES_DIR, 'index.html')

@app.route('/examples/<path:filename>')
def serve_examples_files(filename):
    return send_from_directory(EXAMPLES_DIR, filename)

@app.route('/theme/<path:filename>')
def serve_examples_theme(filename):
    return send_from_directory(os.path.join(EXAMPLES_DIR, 'theme'), filename)

@app.route('/resources/<path:filename>')
def serve_examples_resources(filename):
    return send_from_directory(os.path.join(EXAMPLES_DIR, 'resources'), filename)

@app.route('/examples/data/<path:filename>')
def serve_examples_data(filename):
    return send_from_directory(os.path.join(EXAMPLES_DIR, 'data'), filename)
# ------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('web', 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/api/geocode', methods=['GET'])
def geocode():
    q = request.args.get('q', '').strip()
    city = request.args.get('city', '').strip()
    key = CONFIG.get('AMAP_KEY') or os.environ.get('AMAP_KEY') or ''
    if not q:
        logger.warning('地理编码缺少参数 q')
        return jsonify({'error': 'missing_q'}), 400
    if not key:
        logger.error('AMAP_KEY 未配置')
        return jsonify({'error': 'missing_key'}), 500
    url = 'https://restapi.amap.com/v3/geocode/geo'
    params = {'address': q, 'key': key}
    if city:
        params['city'] = city
    try:
        import requests
        r = requests.get(url, params=params, timeout=7)
        r.raise_for_status()
        j = r.json()
        if j.get('status') != '1':
            logger.error('高德返回错误: %s', j)
            return jsonify({'error': 'amap_error', 'data': j}), 502
        geocodes = j.get('geocodes') or []
        if not geocodes:
            logger.info('地理编码未命中: %s | %s', q, city)
            return jsonify({'error': 'not_found'}), 404
        loc = geocodes[0].get('location', '')
        if not loc:
            logger.info('地理编码无坐标: %s', geocodes[0])
            return jsonify({'error': 'no_location'}), 404
        lon_gcj, lat_gcj = [float(x) for x in loc.split(',')]
        try:
            import numpy as np
            from xyconvert import gcj2wgs
            wgs = gcj2wgs(np.array([[lon_gcj, lat_gcj]]))
            lon_wgs, lat_wgs = float(wgs[0, 0]), float(wgs[0, 1])
            logger.info('地理编码成功(GCJ->WGS): %s -> (%.6f, %.6f) -> (%.6f, %.6f)', q, lon_gcj, lat_gcj, lon_wgs, lat_wgs)
            return jsonify({'lon': lon_wgs, 'lat': lat_wgs, 'provider': 'amap', 'coord_in': 'GCJ-02', 'coord_out': 'WGS84'})
        except Exception as conv_e:
            logger.warning('坐标转换失败，回退使用GCJ-02: %s', conv_e)
            return jsonify({'lon': lon_gcj, 'lat': lat_gcj, 'provider': 'amap', 'coord_out': 'GCJ-02'})
    except Exception as e:
        logger.exception('地理编码异常')
        return jsonify({'error': 'server_error', 'detail': str(e)}), 500

def _proxy_tianditu(T, z, x, y):
    tk = CONFIG.get('TIANDITU_TK') or os.environ.get('TIANDITU_TK') or ''
    if not tk:
        logger.error('TIANDITU_TK 未配置')
        return jsonify({'error': 'missing_tk'}), 500
    s = random.randint(0, 7)
    url = f"https://t{s}.tianditu.gov.cn/DataServer?T={T}&x={x}&y={y}&l={z}&tk={tk}"
    try:
        import requests
        r = requests.get(url, timeout=10)
        ct = r.headers.get('Content-Type', '')
        if not r.ok or not (ct.startswith('image/') or ct == 'application/octet-stream'):
            body = r.text[:200] if hasattr(r, 'text') else ''
            logger.error('天地图瓦片异常: status=%s ct=%s body=%s url=%s', r.status_code, ct, body, url)
            return jsonify({'error': 'tianditu_error', 'status': r.status_code}), 502
        resp = Response(r.content, mimetype=ct or 'image/png')
        resp.headers['Cache-Control'] = 'public, max-age=604800'
        return resp
    except Exception as e:
        logger.exception('瓦片代理异常')
        return jsonify({'error': 'server_error', 'detail': str(e)}), 500

@app.route('/tiles/vec/<int:z>/<int:x>/<int:y>')
def tiles_vec(z, x, y):
    return _proxy_tianditu('vec_w', z, x, y)

@app.route('/tiles/cva/<int:z>/<int:x>/<int:y>')
def tiles_cva(z, x, y):
    return _proxy_tianditu('cva_w', z, x, y)

@app.route('/api/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.get_json(force=True)
        requirements = data.get('requirements', '').strip()
        # 文本权重固定为 1.0
        w_text = 1.0
        # 禁用 SAFE：强制不使用 SAFE 权重
        w_safe = 0.0
        min_site_candidate_num = int(data.get('top_k', 10))
        city = data.get('city', 'guangzhou')
        type_ = data.get('type', 'zh')

        if not requirements:
            logger.warning('推荐请求缺少需求描述')
            return jsonify({"error": "需求描述不能为空"}), 400

        # Load API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            logger.error('OPENAI_API_KEY 未设置')
            return jsonify({"error": "OPENAI_API_KEY 未设置，请在环境变量中配置"}), 400

        # 支持通过环境变量设置自定义 Base URL（如国内代理服务）
        # OpenaiCall 内部也会自动读取 OPENAI_BASE_URL / OPENAI_API_BASE / OPENAI_PROXY_BASE
        proxy = OpenaiCall(api_key=api_key)

        # 使用带交通与价格指标的真实数据CSV（自动生成同名npy）
        dataset_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'data', 'land_transactions_with_coordinates_metrics.csv'))

        selector = SiteSelector(
            user_reqs=requirements,
            city=city,
            min_site_candidate_num=min_site_candidate_num,
            proxy_call=proxy,
            type=type_,
            blend_w_text=w_text,
            blend_w_safe=w_safe,
            enable_safe=False,
            dataset_path=dataset_csv_path
        )

        logger.info('开始生成推荐: city=%s top_k=%s', city, min_site_candidate_num)
        result = selector.solve()
        logger.info('推荐生成完成')
        # result is expected to contain: features (GeoJSON-like), center {lon, lat}, sites list, etc.
        return jsonify(result)

    except Exception as e:
        logger.exception('推荐服务异常')
        return jsonify({"error": f"服务端异常: {str(e)}"}), 500


if __name__ == '__main__':
    # Allow port override via env
    port = int(os.environ.get('PORT', '8000'))
    logger.info('服务启动: port=%d', port)
    app.run(host='0.0.0.0', port=port, debug=True)