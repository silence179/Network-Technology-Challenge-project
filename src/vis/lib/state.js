import * as Cesium from 'cesium';

export const shared = {
    startUtc: Cesium.JulianDate.fromIso8601("2026-03-05T08:00:00Z"),
    state: { showTopology: true, showAnalytics: false, showPerformance: false, currentTarget: null, viewMode: 'OVERVIEW' },
    entityMap: new Map(),
    topologyEvents: [],
    nodeInfoList: [],
    chart2D: null,
    lastEdgesString: "",
    perfChart: null
};
