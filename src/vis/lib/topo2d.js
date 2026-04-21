import * as echarts from 'echarts';
import * as Cesium from 'cesium';
import { shared } from './state.js';
import { getLinkStatus } from './utils.js';

// 辅助函数：通过名称判断节点类型
function getNodeType(name) {
    const upperName = (name || '').toUpperCase();
    if (upperName.includes('GS') || upperName.includes('GROUND')) return 'GS';
    if (upperName.includes('UAV')) return 'UAV';
    return 'SAT'; // 默认为卫星
}

export function update2DTopology(viewer) {
    const container = document.getElementById('topo-2d-container');
    if (!shared.state.showAnalytics || !container) {
        if (shared.chart2D) { shared.chart2D.clear(); shared.lastEdgesString = ""; }
        return;
    }

    // 获取容器宽高，用于计算固定节点的位置
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;
    
    // 辅助函数：生成节点的通用属性（包含固定位置、大小、斥力值等）
    const generateNodeProps = (n, index, connectedNodes) => {
        const type = getNodeType(n.name);
        const isGS = type === 'GS';
        
        // 计算基于连接节点的统计
        const totalNodeCount = connectedNodes.length || 1;
        const gsNodes = connectedNodes.filter(node => getNodeType(node.name) === 'GS');
        const gsCount = gsNodes.length || 1;
        
        let x, y, fixed;
        if (isGS) {
            // 将 GS 固定在底部，并根据数量均匀分布
            const gsIdx = gsNodes.findIndex(gs => gs.id === n.id);
            x = (width / (gsCount + 1)) * (gsIdx + 1);
            y = height - 60; // 距离底部 60px
            fixed = true;    // 开启固定
        } else {
            // 其他节点初始按圆形分布在上方
            const angle = (index / totalNodeCount) * Math.PI * 2;
            const radius = Math.min(width, height) * 0.35;
            x = width / 2 + Math.cos(angle) * radius;
            y = height / 2 - 50 + Math.sin(angle) * radius;
            fixed = false;
        }
        
        return {
            id: String(n.id),
            name: n.name,
            x: x,
            y: y,
            fixed: fixed,
            // 大小区分：GS最大(40)，UAV中等(35)，卫星最小(20)
            symbolSize: isGS ? 40 : (type === 'UAV' ? 35 : 20),
            // value用于映射斥力：GS和UAV斥力大，卫星斥力小
            value: isGS ? 200 : (type === 'UAV' ? 100 : 10),
            itemStyle: { 
                // 颜色区分：当前目标红色，GS绿色，UAV橙色，卫星青色
                color: n.id === shared.state.currentTarget ? '#ff4757' : 
                       (isGS ? '#2ecc71' : type === 'UAV' ? '#f39c12' : '#00f2ff'), 
                shadowBlur: n.id === shared.state.currentTarget ? 15 : 0, 
                shadowColor: '#fff' 
            }
        };
    };

    if (!shared.chart2D) {
        shared.chart2D = echarts.init(container, 'dark');

        // 初始化时使用所有节点计算参数
        const initTotalNodeCount = shared.nodeInfoList.length || 1;
        const baseRepulsion = Math.min(2000, 600 + initTotalNodeCount * 18);
        const baseEdgeLength = Math.min(300, 120 + initTotalNodeCount * 4);
        const baseGravity = 0.15;

        // 初始化时显示所有节点
        const initialNodesWithPos = shared.nodeInfoList.map((n, i) => generateNodeProps(n, i, shared.nodeInfoList));

        shared.chart2D.setOption({
            backgroundColor: 'transparent',
            series: [{
                type: 'graph', 
                layout: 'force', 
                data: initialNodesWithPos, 
                draggable: true, 
                roam: true,
                force: { 
                    // 改为数组：让 ECharts 根据 node.value 动态映射斥力
                    repulsion: [baseRepulsion * 0.3, baseRepulsion * 3], 
                    // 改为数组：让 ECharts 根据 link.value 动态映射边长
                    edgeLength: [baseEdgeLength * 0.5, baseEdgeLength * 2.5], 
                    gravity: baseGravity, 
                    // 取消 initLayout，使用我们手动计算的 x, y 初始位置
                    layoutAnimation: true 
                },
                label: { show: true, position: 'right', color: '#fff', formatter: '{b}' },
                edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 8, lineStyle: { width: 2, curveness: 0.1 }
            }]
        });

        // 拖拽防抖逻辑保持不变
        try {
            let dragTimer = null;
            shared.chart2D.on && shared.chart2D.on('dragstart', params => {
                if (params && params.dataType === 'node') {
                    // 注意这里的 repulsion 也最好保持数组格式，或者直接给个中间值
                    shared.chart2D.setOption({ series: [{ force: { repulsion: [80, baseRepulsion], gravity: 0.5, layoutAnimation: false } }] }, false);
                    if (dragTimer) { clearTimeout(dragTimer); dragTimer = null; }
                }
            });
            shared.chart2D.on && shared.chart2D.on('dragend', params => {
                if (params && params.dataType === 'node') {
                    shared.chart2D.setOption({ series: [{ force: { repulsion: [baseRepulsion * 0.3, baseRepulsion * 3], gravity: baseGravity, layoutAnimation: true } }] }, false);
                    if (dragTimer) clearTimeout(dragTimer);
                    dragTimer = setTimeout(() => {
                        try { shared.chart2D.setOption({ series: [{ force: { layoutAnimation: false } }] }, false); } catch (e) {}
                        dragTimer = null;
                    }, 900);
                }
            });
        } catch (e) { /* ignore */ }
    }

    // 更新连线数据
    const ms = Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, shared.startUtc) * 1000;
    const edges = [];
    const connectedNodeIds = new Set(); // 记录有连接的节点ID
    
    // 性能优化：如果节点数量太多，限制计算量
    const nodeCount = shared.nodeInfoList.length;
    const maxNodesToProcess = 50; // 限制处理的节点数量
    
    // 只处理前 maxNodesToProcess 个节点，避免 O(n²) 复杂度导致性能问题
    const nodesToProcess = nodeCount > maxNodesToProcess ? 
        shared.nodeInfoList.slice(0, maxNodesToProcess) : 
        shared.nodeInfoList;
    
    nodesToProcess.forEach((n1, i) => {
        // 对于每个节点，只检查与后续节点的连接
        shared.nodeInfoList.slice(i + 1).forEach(n2 => {
            const status = getLinkStatus(n1.id, n2.id, ms);
            if (status) {
                const type1 = getNodeType(n1.name);
                const type2 = getNodeType(n2.name);
                // 核心逻辑：卫星之间的连线极短(value小)，涉及GS/UAV的连线长(value大)
                const edgeValue = (type1 === 'SAT' && type2 === 'SAT') ? 10 : 100;
                edges.push({ source: String(n1.id), target: String(n2.id), status, value: edgeValue });
                // 记录有连接的节点
                connectedNodeIds.add(String(n1.id));
                connectedNodeIds.add(String(n2.id));
            }
        });
    });
    
    // 获取当前时间
    const currentTime = viewer.clock.currentTime;
    
    // 过滤节点：只显示有连接的节点，并且当前时间有位置数据的节点
    const connectedNodes = shared.nodeInfoList.filter(n => {
        // 检查是否有连接
        if (!connectedNodeIds.has(String(n.id))) {
            return false;
        }
        
        // 检查当前时间是否有位置数据
        const entity = shared.entityMap.get(n.id);
        if (!entity || !entity.position || !entity.position.getValue) {
            return false;
        }
        
        try {
            const position = entity.position.getValue(currentTime);
            return position !== undefined && position !== null;
        } catch (e) {
            return false;
        }
    });

    const currentEdgesString = JSON.stringify(edges.map(e => e.source + e.target + e.status).sort());
    if (currentEdgesString === shared.lastEdgesString) return;
    shared.lastEdgesString = currentEdgesString;

    // 渲染更新
    shared.chart2D.setOption({
        series: [{
            // 只显示有连接的节点
            data: connectedNodes.map((n, i) => generateNodeProps(n, i, connectedNodes)),
            links: edges.map(e => ({ 
                source: e.source, 
                target: e.target, 
                value: e.value, // 传入控制边长的 value
                label: { show: true, formatter: e.status, color: '#00ff88', fontSize: 10 },
                lineStyle: { color: e.status.includes('up') ? '#00f2ff' : '#f1c40f', opacity: 0.8, width: 2 }
            }))
        }]
    }, false);
}
