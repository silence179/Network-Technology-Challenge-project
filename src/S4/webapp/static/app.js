/* S4 网络仿真引擎 —— 前端交互逻辑 */

let simResults = [];
let simJobId = null;
let currentSort = { field: 'time_ms', asc: true };
let latencyChart = null;
let throughputChart = null;

// === 文件上传处理 ===
document.querySelectorAll('.upload-zone').forEach(zone => {
    const input = zone.querySelector('input[type="file"]');
    const nameSpan = zone.querySelector('.file-name');

    input.addEventListener('change', () => {
        if (input.files.length === 0) {
            nameSpan.textContent = '未选择文件';
        } else if (input.files.length === 1) {
            nameSpan.textContent = input.files[0].name;
        } else {
            nameSpan.textContent = input.files.length + ' 个文件已选';
        }
    });

    // 拖拽上传
    ['dragenter', 'dragover'].forEach(evt => {
        zone.addEventListener(evt, e => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        zone.addEventListener(evt, e => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (evt.type === 'drop') {
                const files = e.dataTransfer.files;
                if (input.multiple) {
                    input.files = files;
                } else if (files.length > 0) {
                    input.files = files;
                }
                input.dispatchEvent(new Event('change'));
            }
        });
    });
});

// === 执行仿真 ===
document.getElementById('btn-run').addEventListener('click', async () => {
    const linksFile = document.getElementById('file-links').files[0];
    const rulesFile = document.getElementById('file-rules').files[0];
    const uavFile = document.getElementById('file-uav').files[0];
    const satFiles = document.getElementById('file-sat').files;
    const scenarioFile = document.getElementById('file-scenario').files[0];
    const mode = document.querySelector('input[name="mode"]:checked').value;

    if (!linksFile || !rulesFile || !uavFile) {
        alert('请上传所有必需文件: links.csv, rules.json, uav.csv');
        return;
    }

    const btn = document.getElementById('btn-run');
    const progressArea = document.getElementById('progress-area');
    const errorArea = document.getElementById('error-area');
    const resultsSection = document.getElementById('results-section');

    btn.disabled = true;
    progressArea.classList.remove('hidden');
    errorArea.classList.add('hidden');
    resultsSection.classList.add('hidden');

    document.getElementById('progress-text').textContent = '仿真运行中...';

    const formData = new FormData();
    formData.append('links', linksFile);
    formData.append('rules', rulesFile);
    formData.append('uav', uavFile);
    formData.append('mode', mode);
    for (const f of satFiles) {
        formData.append('sat', f);
    }
    if (scenarioFile) {
        formData.append('scenario', scenarioFile);
    }

    try {
        const resp = await fetch('/api/simulate', { method: 'POST', body: formData });
        const data = await resp.json();

        if (data.success) {
            simResults = data.results || [];
            simJobId = data.job_id || null;
            renderSummary(data.summary);
            renderTable(simResults);
            renderCharts(simResults);
            resultsSection.classList.remove('hidden');
            document.getElementById('progress-text').textContent = '仿真完成';
        } else {
            showError(data.error || '仿真执行失败');
        }
    } catch (err) {
        showError('网络错误: ' + err.message);
    } finally {
        btn.disabled = false;
        setTimeout(() => progressArea.classList.add('hidden'), 1500);
    }
});

// 显示错误信息
function showError(msg) {
    const area = document.getElementById('error-area');
    const text = document.getElementById('error-text');
    area.classList.remove('hidden');
    text.textContent = msg;
}

// === 摘要统计 ===
function renderSummary(s) {
    if (!s) return;
    document.getElementById('stat-requests').textContent = s.total_requests ?? '-';
    document.getElementById('stat-latency').textContent = s.avg_latency_ms != null ? s.avg_latency_ms.toFixed(1) : '-';
    document.getElementById('stat-throughput').textContent = s.avg_throughput_mbps != null ? s.avg_throughput_mbps.toFixed(1) : '-';
    document.getElementById('stat-success').textContent = s.success_rate != null ? s.success_rate + '%' : '-';
    document.getElementById('stat-data').textContent = s.total_data_mb != null ? s.total_data_mb.toFixed(1) : '-';
    document.getElementById('stat-maxlat').textContent = s.max_latency_ms != null ? s.max_latency_ms.toFixed(1) : '-';
}

// === 结果表格 ===
function renderTable(results) {
    const tbody = document.getElementById('results-body');
    const info = document.getElementById('table-info');
    const filterText = (document.getElementById('table-filter').value || '').toLowerCase();
    const maxRows = 2000;

    let filtered = results;
    if (filterText) {
        filtered = results.filter(r =>
            Object.values(r).some(v => String(v).toLowerCase().includes(filterText))
        );
    }

    // 排序
    const sorted = [...filtered].sort((a, b) => {
        const va = a[currentSort.field];
        const vb = b[currentSort.field];
        const na = isNaN(va) ? va : Number(va);
        const nb = isNaN(vb) ? vb : Number(vb);
        if (na < nb) return currentSort.asc ? -1 : 1;
        if (na > nb) return currentSort.asc ? 1 : -1;
        return 0;
    });

    const display = sorted.slice(0, maxRows);

    tbody.innerHTML = display.map(r => `
        <tr>
            <td>${r.time_ms ?? ''}</td>
            <td>${r.req_id ?? ''}</td>
            <td>${r.node_id ?? ''}</td>
            <td>${r.content_id ?? ''}</td>
            <td>${r.file_size_MB ?? ''}</td>
            <td>${r.algo ?? ''}</td>
            <td>${r.server_node ?? ''}</td>
            <td>${r.latency_ms != null ? Number(r.latency_ms).toFixed(2) : ''}</td>
            <td>${r.throughput_mbps != null ? Number(r.throughput_mbps).toFixed(2) : ''}</td>
            <td>${r.download_time != null ? Number(r.download_time).toFixed(2) : ''}</td>
            <td>${r.http_code ?? ''}</td>
        </tr>
    `).join('');

    info.textContent = `显示 ${display.length} / ${results.length} 条结果` +
        (filtered.length !== results.length ? `（筛选自 ${results.length} 条）` : '') +
        (sorted.length > maxRows ? `，限制最多 ${maxRows} 行` : '');
}

// 表格点击排序
document.getElementById('results-table').addEventListener('click', e => {
    const th = e.target.closest('th');
    if (!th) return;
    const field = th.dataset.sort;
    if (!field) return;

    if (currentSort.field === field) {
        currentSort.asc = !currentSort.asc;
    } else {
        currentSort.field = field;
        currentSort.asc = true;
    }

    document.querySelectorAll('#results-table th').forEach(h => {
        h.classList.remove('sorted-asc', 'sorted-desc');
    });
    th.classList.add(currentSort.asc ? 'sorted-asc' : 'sorted-desc');

    renderTable(simResults);
});

// 表格筛选
document.getElementById('table-filter').addEventListener('input', () => {
    renderTable(simResults);
});

// 下载 CSV
document.getElementById('btn-download').addEventListener('click', () => {
    if (!simJobId) return;
    window.location.href = '/api/download/' + simJobId;
});

// === 图表绘制 ===
function renderCharts(results) {
    if (!results || results.length === 0) return;

    if (latencyChart) latencyChart.destroy();
    if (throughputChart) throughputChart.destroy();

    // 对大量数据采样，避免散点图过密
    const sampleStep = Math.max(1, Math.floor(results.length / 500));
    const latencyData = [];
    const timeLabels = [];

    for (let i = 0; i < results.length; i += sampleStep) {
        const r = results[i];
        const t = Number(r.time_ms);
        const l = Number(r.latency_ms);
        if (!isNaN(t) && !isNaN(l)) {
            timeLabels.push(t);
            latencyData.push(l);
        }
    }

    // 延迟时间散点图
    const ctx1 = document.getElementById('chart-latency').getContext('2d');
    latencyChart = new Chart(ctx1, {
        type: 'scatter',
        data: {
            datasets: [{
                label: '延迟 (ms)',
                data: timeLabels.map((t, i) => ({ x: t / 1000, y: latencyData[i] })),
                backgroundColor: 'rgba(88, 166, 255, 0.4)',
                borderColor: 'rgba(88, 166, 255, 0.7)',
                pointRadius: 2,
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: '时间 (s)', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                y: { title: { display: true, text: '延迟 (ms)', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // 吞吐量分布直方图
    const tpData = results
        .map(r => Number(r.throughput_mbps))
        .filter(v => !isNaN(v) && v > 0 && v < 500);

    const binCount = 40;
    const minTp = Math.min(...tpData);
    const maxTp = Math.max(...tpData);
    const binWidth = (maxTp - minTp) / binCount || 1;
    const bins = new Array(binCount).fill(0);
    tpData.forEach(v => {
        const idx = Math.min(Math.floor((v - minTp) / binWidth), binCount - 1);
        bins[idx]++;
    });
    const binLabels = bins.map((_, i) => (minTp + i * binWidth).toFixed(1));

    const ctx2 = document.getElementById('chart-throughput').getContext('2d');
    throughputChart = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: '频次',
                data: bins,
                backgroundColor: 'rgba(63, 185, 80, 0.5)',
                borderColor: 'rgba(63, 185, 80, 0.8)',
                borderWidth: 1,
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: '吞吐量 (Mbps)', color: '#8b949e' }, ticks: { color: '#8b949e', maxTicksLimit: 15, maxRotation: 45 }, grid: { color: '#21262d' } },
                y: { title: { display: true, text: '频次', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}
