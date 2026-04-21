import { shared } from './state.js';

// 缓存对象，用于存储按key和时间索引的事件
let topologyCache = null;
let lastCacheTime = null;

// 构建缓存
function buildTopologyCache() {
    if (!shared.topologyEvents || shared.topologyEvents.length === 0) {
        topologyCache = new Map();
        return;
    }
    
    const cache = new Map();
    
    // 按key分组事件，并按时间排序（升序）
    shared.topologyEvents.forEach(event => {
        if (!cache.has(event.key)) {
            cache.set(event.key, []);
        }
        cache.get(event.key).push(event);
    });
    
    // 对每个key的事件按时间升序排序，便于二分查找
    for (const [key, events] of cache.entries()) {
        events.sort((a, b) => a.time - b.time);
    }
    
    topologyCache = cache;
    lastCacheTime = Date.now();
}

// 二分查找：在已排序的数组中查找最后一个时间 <= targetTime 的事件
function findLastEventByTime(events, targetTime) {
    let left = 0;
    let right = events.length - 1;
    let result = -1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (events[mid].time <= targetTime) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result >= 0 ? events[result] : null;
}

export function getLinkStatus(idA, idB, currentTimeMs) {
    // 如果缓存未构建或拓扑事件发生变化，重新构建缓存
    if (!topologyCache || (shared.topologyEvents.length > 0 && !lastCacheTime)) {
        buildTopologyCache();
    }
    
    const key = [String(idA), String(idB)].sort().join('---');
    const events = topologyCache.get(key);
    
    if (!events || events.length === 0) {
        return null;
    }
    
    // 使用二分查找快速找到对应时间的事件
    const lastEvent = findLastEventByTime(events, currentTimeMs);
    return lastEvent ? lastEvent.status : null;
}

// 导出清理缓存的函数（如果需要）
export function clearTopologyCache() {
    topologyCache = null;
    lastCacheTime = null;
}
