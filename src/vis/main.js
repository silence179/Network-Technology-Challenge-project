import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import { shared } from './lib/state.js';
import { initSystem } from './lib/initSystem.js';
import { update2DTopology } from './lib/topo2d.js';
import { updatePerformanceChart, startPerformanceUpdates, stopPerformanceUpdates } from './lib/perf2d.js';

Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlNjdhNjdjZC1mMjA0LTQwMWEtYTcwYi02MTA5YWY5ZTZhYzEiLCJpZCI6Mzg4MDQ1LCJpYXQiOjE3NzA0Mzc0MTR9.Dfqi_zBqKJV_Yia-9waxWRMQ5VkYP4IAkQin7t5vVao';

const viewer = new Cesium.Viewer('app', {
    timeline: false,
    terrainProvider: null,
    baseLayerPicker: false,
    shouldAnimate: true,
    selectionIndicator: false,
    infoBox: false,
    navigationHelpButton: false,
    sceneModePicker: false,
    homeButton: false,
    geocoder: false
});

window.toggleOption = (type) => {
    if (type === 'ANA') {
        shared.state.showAnalytics = !shared.state.showAnalytics;
        const anaView = document.getElementById('analytics-view');
        const appView = document.getElementById('app');
        const right = document.getElementById('main-chart-right');
        if (shared.state.showAnalytics) {
            anaView.style.height = '40%';
            appView.style.height = '60%';
            if (right) right.style.display = 'none';
            setTimeout(() => { shared.chart2D?.resize(); update2DTopology(viewer); }, 400);
        } else {
            anaView.style.height = '0';
            appView.style.height = '100%';
            if (right) right.style.display = 'block';
        }
    } else if (type === 'PERF') {
        shared.state.showPerformance = !shared.state.showPerformance;
        const perfView = document.getElementById('performance-view');
        const appView = document.getElementById('app');
        if (shared.state.showPerformance) {
            perfView.style.height = '40%';
            appView.style.height = '60%';
            // 更新性能图表
            setTimeout(() => { 
                updatePerformanceChart(); 
                shared.perfChart?.resize();
                startPerformanceUpdates(viewer);
            }, 400);
        } else {
            perfView.style.height = '0';
            appView.style.height = '100%';
            stopPerformanceUpdates();
        }
    }
};

// changeView no longer used (全局预览/跟踪已移除)
window.changeView = (mode) => {};

// 选择框现在仅作为视图模式开关：保存当前目标但不自动跟踪
window.selectTarget = (id) => {
    shared.state.currentTarget = id || null;
    if (!id) {
        viewer.trackedEntity = undefined;
        try { viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY); } catch (e) { /* ignore */ }
        try { viewer.zoomTo(viewer.entities); } catch (e) { /* ignore */ }
        return;
    }
    const target = shared.entityMap.get(id);
    if (target) {
        viewer.trackedEntity = target;
        const defaultViewFrom = new Cesium.Cartesian3(-800.0, -800.0, 450.0);
        try { viewer.trackedEntity.viewFrom = target.viewFrom || defaultViewFrom; } catch (e) { /* ignore */ }
    }
};

initSystem(viewer);

// 初始化性能图表更新
// 注意：这里不立即启动更新，只在窗口打开时启动
//