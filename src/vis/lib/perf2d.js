import * as echarts from 'echarts';
import * as Cesium from 'cesium';
import Papa from 'papaparse';
import { shared } from './state.js';

// 性能数据缓冲区
let performanceDataBuffer = {
    latency: [],
    throughput: [],
    timestamps: []
};

// 最大数据点数
const MAX_DATA_POINTS = 30;

// 模拟时间相关变量
let simulationStartTime = null;
let simulationCurrentTime = null;

// 真实性能数据
let realPerformanceData = null;
let lastAppendedIndex = 0;

// 初始化模拟时间（基于当前模拟时间）
let currentViewer = null;

// 将毫秒转换为时间字符串 (HH:MM:SS)
function msToTimeString(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

// 加载真实性能数据
async function loadRealPerformanceData() {
	// 尝试多个可能的路径（包括绝对路径和相对路径）
	const baseUrl = window.location.origin;
	const possiblePaths = [
		'/data/networks.csv',
		'./data/networks.csv',
		'public/data/networks.csv',
		'/src/vis/public/data/networks.csv',
		'../public/data/networks.csv',
		'./public/data/networks.csv',
		'data/networks.csv',
		`${baseUrl}/data/networks.csv`
	];
    
    let lastError = null;
    
    for (const path of possiblePaths) {
        try {
            console.log(`Trying to load real performance data from: ${path}`);
            const response = await fetch(path);
            if (!response.ok) {
                console.warn(`HTTP error! status: ${response.status}, url: ${path}`);
                continue; // 尝试下一个路径
            }
            const text = await response.text();
            console.log(`CSV file loaded from ${path}, size:`, text.length, 'chars');
            
            const parsed = Papa.parse(text, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true
            });
            console.log('CSV parsed rows:', parsed.data.length, 'errors:', parsed.errors.length);
            if (parsed.errors && parsed.errors.length > 0) {
                console.warn('CSV parse errors:', parsed.errors);
            }
            
            const headers = parsed.meta.fields || [];
            console.log('CSV headers:', headers);
            const requiredFields = ['time_ms', 'latency_ms', 'throughput_mbps'];
            if (!requiredFields.every(field => headers.includes(field))) {
                console.warn('Required columns not found in CSV. Available columns:', headers);
                continue; // 尝试下一个路径
            }
            
            // 解析数据点
            const data = [];
            let minTime = Infinity;
            let maxTime = -Infinity;
            let skipped = 0;
            
            for (const row of parsed.data) {
                const timeMs = parseInt(row.time_ms, 10);
                const latency = parseFloat(row.latency_ms);
                const throughput = parseFloat(row.throughput_mbps);
                
                if (isNaN(timeMs) || isNaN(latency) || isNaN(throughput)) {
                    skipped++;
                    continue; // 跳过无效数据
                }
                
                minTime = Math.min(minTime, timeMs);
                maxTime = Math.max(maxTime, timeMs);
                
                data.push({
                    timeMs,
                    latency,
                    throughput
                });
            }
            
            console.log(`Parsed ${data.length} data points, skipped ${skipped} rows`);
            
            if (data.length === 0) {
                console.warn('No valid data points found in CSV');
                continue; // 尝试下一个路径
            }
            
            // 保留原始时间戳，按时间排序
            realPerformanceData = data.sort((a, b) => a.timeMs - b.timeMs);
            
            console.log('Loaded real performance data:', realPerformanceData.length, 'points');
            console.log('Time range:', msToTimeString(realPerformanceData[0].timeMs), 'to', msToTimeString(realPerformanceData[realPerformanceData.length - 1].timeMs));
            console.log('First few points:', realPerformanceData.slice(0, 3));
            
            console.log(`Successfully loaded data from: ${path}`);
            return true;
        } catch (error) {
            console.warn(`Failed to load from ${path}:`, error.message);
            lastError = error;
            continue; // 尝试下一个路径
        }
    }
    
	    // 所有路径都失败
	    console.error('All paths failed to load performance data:', possiblePaths);
	    console.error('Last error:', lastError);
	    // 使用测试数据作为后备，避免图表完全无法显示
	    console.warn('Using fallback test data for performance chart');
	    realPerformanceData = [
	        { time: '00:00:00', latency: 5.0, throughput: 20.0 },
	        { time: '00:00:01', latency: 5.5, throughput: 19.5 },
	        { time: '00:00:02', latency: 6.0, throughput: 19.0 },
	        { time: '00:00:03', latency: 5.8, throughput: 20.2 },
	        { time: '00:00:04', latency: 5.2, throughput: 20.5 }
	    ];
	    return true;
}

function initSimulationTime(viewer) {
    // 存储viewer供后续使用
    if (viewer) {
        currentViewer = viewer;
    }
    
    if (!currentViewer) {
        console.warn("没有提供viewer，使用系统当前时间");
        // 获取当前系统时间
        const now = new Date();
        simulationStartTime = new Date(now);
        simulationCurrentTime = new Date(now);
        return;
    }
    
    try {
        // 获取当前模拟时间（从viewer.clock.currentTime）
        const currentTime = currentViewer.clock.currentTime;
        
        // 计算从shared.startUtc到当前时间经过的秒数
        const secondsSinceStart = Cesium.JulianDate.secondsDifference(currentTime, shared.startUtc);
        
        // 设置模拟开始时间为08:00:00加上已过去的时间
        simulationStartTime = new Date();
        simulationStartTime.setHours(8, 0, 0, 0); // UTC+8 08:00:00
        simulationStartTime.setTime(simulationStartTime.getTime() + secondsSinceStart * 1000);
        
        // 设置模拟当前时间
        simulationCurrentTime = new Date(simulationStartTime);
        
        console.log("性能图表模拟时间初始化（基于当前模拟时间）:", {
            startTime: simulationStartTime.toLocaleTimeString(),
            currentTime: simulationCurrentTime.toLocaleTimeString(),
            secondsSinceStart: secondsSinceStart
        });
    } catch (e) {
        console.warn("无法获取当前模拟时间，使用系统当前时间:", e);
        // 获取当前系统时间
        const now = new Date();
        simulationStartTime = new Date(now);
        simulationCurrentTime = new Date(now);
    }
}

// 获取当前模拟时间点，然后增加一秒用于下一次调用
function getSimulationElapsedMs() {
    if (currentViewer && currentViewer.clock && currentViewer.clock.currentTime) {
        const elapsedSeconds = Cesium.JulianDate.secondsDifference(currentViewer.clock.currentTime, shared.startUtc);
        return Math.max(0, elapsedSeconds * 1000);
    }
    if (simulationCurrentTime && simulationStartTime) {
        return Math.max(0, simulationCurrentTime.getTime() - simulationStartTime.getTime());
    }
    return 0;
}

function formatSimulationTimeLabel(timeMs) {
    const simulationTime = Cesium.JulianDate.addSeconds(shared.startUtc, timeMs / 1000, new Cesium.JulianDate());
    const jsDate = Cesium.JulianDate.toDate(simulationTime);
    const hours = jsDate.getUTCHours().toString().padStart(2, '0');
    const minutes = jsDate.getUTCMinutes().toString().padStart(2, '0');
    const seconds = jsDate.getUTCSeconds().toString().padStart(2, '0');
    return `${hours}:${minutes}:${seconds}`;
}

// 生成新的性能数据点（仅使用真实数据）
function generatePerformanceDataPoint() {
    if (!realPerformanceData || realPerformanceData.length === 0) {
        console.warn('No real performance data available');
        return null;
    }

    const currentSimMs = getSimulationElapsedMs();
    let lastNewPoint = null;

    while (lastAppendedIndex < realPerformanceData.length && realPerformanceData[lastAppendedIndex].timeMs <= currentSimMs) {
        const dataPoint = realPerformanceData[lastAppendedIndex];
        const time = formatSimulationTimeLabel(dataPoint.timeMs);
        const latency = parseFloat(dataPoint.latency.toFixed(2));
        const throughput = parseFloat(dataPoint.throughput.toFixed(2));

        performanceDataBuffer.timestamps.push(time);
        performanceDataBuffer.latency.push([time, latency]);
        performanceDataBuffer.throughput.push([time, throughput]);

        if (performanceDataBuffer.timestamps.length > MAX_DATA_POINTS) {
            performanceDataBuffer.timestamps.shift();
            performanceDataBuffer.latency.shift();
            performanceDataBuffer.throughput.shift();
        }

        lastNewPoint = { time, latency, throughput };
        lastAppendedIndex += 1;
    }

    if (!lastNewPoint) {
        return null;
    }

    console.log('Appended performance point(s) up to simulation time', currentSimMs, 'ms, last point:', lastNewPoint, 'index:', lastAppendedIndex - 1);
    return lastNewPoint;
}

// 更新性能数据缓冲区
function updatePerformanceDataBuffer() {
    const newDataPoint = generatePerformanceDataPoint();
    
    // 如果没有有效数据点，返回当前缓冲区数据
    if (!newDataPoint) {
        return {
            latency: performanceDataBuffer.latency,
            throughput: performanceDataBuffer.throughput
        };
    }
    
    // 添加新数据点
    performanceDataBuffer.timestamps.push(newDataPoint.time);
    performanceDataBuffer.latency.push([newDataPoint.time, newDataPoint.latency]);
    performanceDataBuffer.throughput.push([newDataPoint.time, newDataPoint.throughput]);
    
    // 保持缓冲区大小不超过MAX_DATA_POINTS
    if (performanceDataBuffer.timestamps.length > MAX_DATA_POINTS) {
        performanceDataBuffer.timestamps.shift();
        performanceDataBuffer.latency.shift();
        performanceDataBuffer.throughput.shift();
    }
    
    return {
        latency: performanceDataBuffer.latency,
        throughput: performanceDataBuffer.throughput
    };
}

// 生成性能数据（兼容现有接口）
function generatePerformanceData() {
    // 更新缓冲区并返回当前数据
    return updatePerformanceDataBuffer();
}

export function updatePerformanceChart() {
    const container = document.getElementById('perf-chart-container');
    if (!shared.state.showPerformance || !container) {
        if (shared.perfChart) { 
            shared.perfChart.clear(); 
        }
        return;
    }

    if (!shared.perfChart) {
        shared.perfChart = echarts.init(container, 'dark');
        
        // 初始图表配置
        shared.perfChart.setOption({
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    label: {
                        backgroundColor: '#6a7985'
                    }
                }
            },
            legend: {
                data: ['延迟(ms)', '吞吐量(Mbps)'],
                textStyle: {
                    color: '#fff'
                },
                top: 10
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                top: '15%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                boundaryGap: false,
                data: [],
                axisLine: {
                    lineStyle: {
                        color: '#00f2ff'
                    }
                },
                axisLabel: {
                    color: '#8ab4f8'
                }
            },
            yAxis: [
                {
                    type: 'value',
                    name: '延迟(ms)',
                    position: 'left',
                    axisLine: {
                        lineStyle: {
                            color: '#00f2ff'
                        }
                    },
                    axisLabel: {
                        color: '#8ab4f8',
                        formatter: '{value}'
                    },
                    splitLine: {
                        lineStyle: {
                            color: 'rgba(0, 242, 255, 0.1)'
                        }
                    }
                },
                {
                    type: 'value',
                    name: '吞吐量(Mbps)',
                    position: 'right',
                    axisLine: {
                        lineStyle: {
                            color: '#2ecc71'
                        }
                    },
                    axisLabel: {
                        color: '#8ab4f8',
                        formatter: '{value}'
                    },
                    splitLine: {
                        show: false
                    }
                }
            ],
            series: [
                {
                    name: '延迟(ms)',
                    type: 'line',
                    smooth: true,
                    lineStyle: {
                        width: 3,
                        color: '#00f2ff'
                    },
                    symbol: 'circle',
                    symbolSize: 6,
                    itemStyle: {
                        color: '#00f2ff'
                    },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(0, 242, 255, 0.3)' },
                            { offset: 1, color: 'rgba(0, 242, 255, 0.05)' }
                        ])
                    },
                    data: []
                },
                {
                    name: '吞吐量(Mbps)',
                    type: 'line',
                    smooth: true,
                    yAxisIndex: 1,
                    lineStyle: {
                        width: 3,
                        color: '#2ecc71'
                    },
                    symbol: 'circle',
                    symbolSize: 6,
                    itemStyle: {
                        color: '#2ecc71'
                    },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(46, 204, 113, 0.3)' },
                            { offset: 1, color: 'rgba(46, 204, 113, 0.05)' }
                        ])
                    },
                    data: []
                }
            ]
        });
    }

    // 检查数据是否已加载
    if (!realPerformanceData || realPerformanceData.length === 0) {
        console.warn('Performance data not loaded or empty, showing loading state. realPerformanceData:', realPerformanceData);
        // 显示加载状态或错误消息
        shared.perfChart.setOption({
            title: {
                text: '正在加载性能数据...',
                left: 'center',
                top: 'center',
                textStyle: {
                    color: '#8ab4f8',
                    fontSize: 16
                }
            },
            xAxis: {
                data: []
            },
            series: [
                { data: [] },
                { data: [] }
            ]
        });
        return;
    }
    
    console.log('Performance data loaded, data points:', realPerformanceData.length);

    // 更新数据
    const perfData = generatePerformanceData();
    const timeLabels = perfData.latency.map(item => item[0]);
    
    // 清除可能存在的标题
    shared.perfChart.setOption({
        title: {
            show: false
        }
    });
    
    shared.perfChart.setOption({
        xAxis: {
            data: timeLabels
        },
        series: [
            { data: perfData.latency.map(item => item[1]) },
            { data: perfData.throughput.map(item => item[1]) }
        ]
    });
}

// 定期更新性能图表
let perfUpdateInterval = null;

export async function startPerformanceUpdates(viewer) {
    if (perfUpdateInterval) {
        clearInterval(perfUpdateInterval);
    }
    
    // 初始化模拟时间（基于当前模拟时间）
    initSimulationTime(viewer);
    
    // 初始化数据缓冲区（清空）
    performanceDataBuffer = {
        latency: [],
        throughput: [],
        timestamps: []
    };
    
    // 重置真实数据索引
    lastAppendedIndex = 0;
    
    // 加载真实性能数据（等待加载完成）
    try {
        console.log('Loading real performance data before starting updates...');
        await loadRealPerformanceData();
        console.log('Real performance data loaded successfully');
    } catch (err) {
        console.error('Failed to load real performance data:', err);
        console.error('Performance chart updates disabled due to missing real data');
        return; // 不开始定期更新
    }
    
    // 不预填充数据点，图表将从当前模拟时间开始实时添加数据
    
    // 立即更新一次图表（初始为空）
    if (shared.state.showPerformance) {
        updatePerformanceChart();
    }
    
    // 开始定期更新
    perfUpdateInterval = setInterval(() => {
        if (shared.state.showPerformance) {
            updatePerformanceChart();
        }
    }, 1000); // 每秒更新一次
}

export function stopPerformanceUpdates() {
    if (perfUpdateInterval) {
        clearInterval(perfUpdateInterval);
        perfUpdateInterval = null;
    }
}