"""S4 网络仿真引擎 —— Flask Web 应用服务端。"""

import os
import sys
import csv
import io
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, Response

_S4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _S4_DIR not in sys.path:
    sys.path.insert(0, _S4_DIR)

from runner import run_simulation

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 最大上传 256MB

# 内存存储仿真结果，供下载接口使用
_results_store = {}


@app.route('/')
def index():
    """返回主页面。"""
    return render_template('index.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """接收用户上传的文件与模式选择，执行仿真并返回结果 JSON。

    必需文件：links.csv、rules.json、uav.csv
    可选文件：sat*.csv（卫星节点）、scenario.py（自定义场景）
    """
    links_file = request.files.get('links')
    rules_file = request.files.get('rules')
    uav_file = request.files.get('uav')
    sat_files = request.files.getlist('sat')
    scenario_file = request.files.get('scenario')

    mode = request.form.get('mode', 'networkx')

    # 检查必需文件
    if not links_file or not rules_file or not uav_file:
        missing = []
        if not links_file:
            missing.append('links.csv')
        if not rules_file:
            missing.append('rules.json')
        if not uav_file:
            missing.append('uav.csv')
        return jsonify({'success': False,
                        'error': f'缺少必需文件: {", ".join(missing)}'}), 400

    try:
        links_text = links_file.read().decode('utf-8')
        rules_text = rules_file.read().decode('utf-8')
        uav_text = uav_file.read().decode('utf-8')

        sat_list = []
        for sf in sat_files:
            if sf.filename:
                sat_list.append((sf.filename, sf.read().decode('utf-8')))

        scenario_text = None
        if scenario_file and scenario_file.filename:
            scenario_text = scenario_file.read().decode('utf-8')

    except Exception as e:
        return jsonify({'success': False,
                        'error': f'文件读取失败: {e}'}), 400

    result = run_simulation(links_text, rules_text, uav_text, sat_list, mode=mode,
                            scenario_text=scenario_text)

    # 成功时生成 job_id 并缓存结果
    if result.get('success'):
        job_id = str(uuid.uuid4())[:8]
        _results_store[job_id] = result
        result['job_id'] = job_id

    return jsonify(result)


@app.route('/api/download/<job_id>', methods=['GET'])
def download_csv(job_id):
    """根据 job_id 下载对应的仿真结果 CSV 文件。"""
    stored = _results_store.get(job_id)
    if not stored:
        return jsonify({'success': False, 'error': '任务不存在'}), 404

    results = stored.get('results', [])
    if not results:
        return jsonify({'success': False, 'error': '无结果可下载'}), 404

    output = io.StringIO()
    fieldnames = ['time_ms', 'req_id', 'node_id', 'content_id', 'file_size_MB',
                  'algo', 'path', 'server_node', 'latency_ms', 'throughput_mbps',
                  'http_code', 'cache_status', 'download_time']
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for row in results:
        writer.writerow(row)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f's4_simulation_{timestamp}.csv'
    )


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8081)
