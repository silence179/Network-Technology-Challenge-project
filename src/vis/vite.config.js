import { defineConfig } from 'vite';
import cesium from 'vite-plugin-cesium';
import { resolve, join } from 'path';
import { exec, spawn } from 'child_process';
import { writeFile, mkdir, readdir, unlink } from 'fs/promises';
import { existsSync, readdirSync, statSync } from 'fs';
import { tmpdir } from 'os';

const DATA_DIR = resolve(__dirname, 'public/data');
const SAT_TRACE_DIR = join(DATA_DIR, 'sat_trace');
const UAV_TRACE_DIR = join(DATA_DIR, 'uav_trace');
const TOPO_DIR = join(DATA_DIR, 'topology_links');
const NETWORKS_DIR = join(DATA_DIR, 'networks');
const S1_SCRIPT = resolve(__dirname, '..', 'S1', 'main_s1.py');
const S1_DIR = resolve(__dirname, '..', 'S1');
const S1_CONFIG_FILE = join(tmpdir(), 's1_config.json');
const S2_SCRIPT = resolve(__dirname, '..', 'S2', 'main_s2.py');
const S2_DIR = resolve(__dirname, '..', 'S2');
const S2_CONFIG_FILE = join(tmpdir(), 's2_config.json');
const S3_SCRIPT = resolve(__dirname, '..', 'S3', 'main_s3.py');
const S3_DIR = resolve(__dirname, '..', 'S3');
const S3_CONFIG_FILE = join(tmpdir(), 's3_config.json');
const S3_TRACES_DIR = resolve(__dirname, '..', 'S3', 'traces');
const S3_OUTPUTS_DIR = resolve(__dirname, '..', 'S3', 'outputs');
const S4_SCRIPT = resolve(__dirname, '..', 'S4', 'main_s4.py');
const S4_DIR = resolve(__dirname, '..', 'S4');
const S4_CONFIG_FILE = join(tmpdir(), 's4_config.json');

// 生成任务状态（内存中）
let satGenStatus = { running: false, startTime: null, endTime: null, error: null };
let uavGenStatus = { running: false, startTime: null, endTime: null, error: null };
let s3GenStatus = { running: false, startTime: null, endTime: null, error: null };
let s4GenStatus = { running: false, startTime: null, endTime: null, error: null };

function trajectoryApiPlugin() {
  return {
    name: 'trajectory-api',
    configureServer(server) {
      // POST /api/open-folder — 在资源管理器中打开文件夹
      server.middlewares.use('/api/open-folder', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }

        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
          try {
            const { folder } = JSON.parse(body);
            const dirMap = { sat: SAT_TRACE_DIR, uav: UAV_TRACE_DIR, topo: TOPO_DIR, networks: NETWORKS_DIR };
            const targetPath = dirMap[folder];
            const cmd = process.platform === 'win32'
              ? `start "" "${targetPath}"`
              : process.platform === 'darwin'
                ? `open "${targetPath}"`
                : `xdg-open "${targetPath}"`;
            exec(cmd, (err) => {
              if (err) { res.statusCode = 500; res.end(JSON.stringify({ error: err.message })); }
              else { res.statusCode = 200; res.end(JSON.stringify({ ok: true, path: targetPath })); }
            });
          } catch (e) {
            res.statusCode = 400; res.end(JSON.stringify({ error: e.message }));
          }
        });
      });

      // POST /api/import-file — 保存上传的 CSV 到目标文件夹
      server.middlewares.use('/api/import-file', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }

        const chunks = [];
        req.on('data', chunk => { chunks.push(chunk); });
        req.on('end', async () => {
          try {
            const boundary = req.headers['content-type']?.match(/boundary=(.+)/)?.[1];
            if (!boundary) { res.statusCode = 400; res.end('no boundary'); return; }

            const raw = Buffer.concat(chunks).toString('binary');
            const parts = raw.split(`--${boundary}`);
            let targetFolder = '';
            let fileName = '';
            let fileData = null;

            for (const part of parts) {
              if (part.includes('name="folder"')) {
                targetFolder = part.split('\r\n\r\n')[1]?.trim();
              }
              if (part.includes('name="file"') && part.includes('filename="')) {
                const headerEnd = part.indexOf('\r\n\r\n');
                const header = part.substring(0, headerEnd);
                const fnMatch = header.match(/filename="(.+?)"/);
                fileName = fnMatch ? fnMatch[1] : 'unknown.csv';
                // file binary data starts after \r\n\r\n, ends before trailing \r\n
                const dataStart = headerEnd + 4;
                let dataEnd = part.lastIndexOf('\r\n');
                if (dataEnd <= dataStart) dataEnd = part.length;
                fileData = part.substring(dataStart, dataEnd);
              }
            }

            if (!targetFolder || !fileName || fileData === null) {
              res.statusCode = 400; res.end('missing fields');
              return;
            }

            const destMap = { sat: SAT_TRACE_DIR, uav: UAV_TRACE_DIR, topo: TOPO_DIR, networks: NETWORKS_DIR };
            const destDir = destMap[targetFolder];
            if (!existsSync(destDir)) await mkdir(destDir, { recursive: true });

            const destPath = join(destDir, fileName);
            await writeFile(destPath, fileData, 'binary');

            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true, path: destPath, name: fileName }));
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: e.message }));
          }
        });
      });

      // GET /api/list-files?folder=sat|uav|topo|networks — 列出目录中的 CSV 文件
      server.middlewares.use('/api/list-files', async (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        try {
          const url = new URL(req.url, 'http://localhost');
          const folder = url.searchParams.get('folder') || 'sat';
          const dirMap = { sat: SAT_TRACE_DIR, uav: UAV_TRACE_DIR, topo: TOPO_DIR, networks: NETWORKS_DIR };
          const dir = dirMap[folder];
          if (!dir || !existsSync(dir)) { res.end('[]'); return; }
          const files = await readdir(dir);
          const csvFiles = files.filter(f => f.endsWith('.csv')).sort();
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(csvFiles));
        } catch (e) {
          res.statusCode = 500; res.end(JSON.stringify({ error: e.message }));
        }
      });

      // POST /api/generate-sat-traces — 启动 S1 模块生成卫星轨迹
      server.middlewares.use('/api/generate-sat-traces', async (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }

        if (satGenStatus.running) {
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, message: '已有生成任务正在运行中' }));
          return;
        }

        // 读取用户配置参数
        let config = {};
        try {
          const chunks = [];
          for await (const chunk of req) { chunks.push(chunk); }
          const body = Buffer.concat(chunks).toString();
          if (body) config = JSON.parse(body);
        } catch (e) { /* 使用默认值 */ }

        satGenStatus = { running: true, startTime: Date.now(), endTime: null, error: null };

        // 先清空 vis 前端 sat_trace 目录中的旧文件
        try {
          if (existsSync(SAT_TRACE_DIR)) {
            const files = await readdir(SAT_TRACE_DIR);
            await Promise.all(files.map(f => unlink(join(SAT_TRACE_DIR, f))));
            console.log(`[S1] 已清空 vis sat_trace 目录 (${files.length} 个文件)`);
          }
        } catch (e) {
          console.warn('[S1] 清空目录失败:', e.message);
        }

        // 将用户配置写入临时 JSON 文件，通过环境变量传递给 Python
        const configPayload = {};
        if (config.duration) configPayload.duration = config.duration;
        if (config.obsLat != null) configPayload.obs_lat = config.obsLat;
        if (config.obsLon != null) configPayload.obs_lon = config.obsLon;
        if (config.obsEle != null) configPayload.obs_ele = config.obsEle;
        if (config.reselectCount) configPayload.reselect_count = config.reselectCount;
        if (config.minAlt != null) configPayload.min_alt = config.minAlt;
        if (config.maxDist) configPayload.max_dist = config.maxDist;
        if (config.chunkDuration) configPayload.chunk_duration = config.chunkDuration;

        await writeFile(S1_CONFIG_FILE, JSON.stringify(configPayload, null, 2), 'utf-8');

        const pyCmd = process.platform === 'win32' ? 'python' : 'python3';
        const proc = spawn(pyCmd, [S1_SCRIPT], {
          cwd: S1_DIR,
          shell: false,
          env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            PYTHONUTF8: '1',
            S1_CONFIG_FILE: S1_CONFIG_FILE
          }
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => { stdout += data.toString(); });
        proc.stderr.on('data', (data) => { stderr += data.toString(); });

        proc.on('close', (code) => {
          satGenStatus.running = false;
          satGenStatus.endTime = Date.now();
          if (code !== 0) {
            satGenStatus.error = stderr || stdout || `exit code ${code}`;
          }
          console.log(`[S1] 卫星轨迹生成完成, exit=${code}, stdout=${stdout.slice(0,200)}`);
        });

        proc.on('error', (err) => {
          satGenStatus.running = false;
          satGenStatus.endTime = Date.now();
          satGenStatus.error = err.message;
        });

        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ok: true, message: '卫星轨迹生成任务已启动' }));
      });

      // GET /api/generate-sat-status — 查询卫星生成任务状态
      server.middlewares.use('/api/generate-sat-status', (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(satGenStatus));
      });

      // POST /api/generate-uav-traces — 启动 S2 模块生成无人机轨迹
      server.middlewares.use('/api/generate-uav-traces', async (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }

        if (uavGenStatus.running) {
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, message: '已有生成任务正在运行中' }));
          return;
        }

        let config = {};
        try {
          const chunks = [];
          for await (const chunk of req) { chunks.push(chunk); }
          const body = Buffer.concat(chunks).toString();
          if (body) config = JSON.parse(body);
        } catch (e) { /* 使用默认值 */ }

        uavGenStatus = { running: true, startTime: Date.now(), endTime: null, error: null };

        // 清空 vis 前端 uav_trace 目录中的旧文件
        try {
          if (existsSync(UAV_TRACE_DIR)) {
            const files = await readdir(UAV_TRACE_DIR);
            await Promise.all(files.map(f => unlink(join(UAV_TRACE_DIR, f))));
            console.log(`[S2] 已清空 vis uav_trace 目录 (${files.length} 个文件)`);
          }
        } catch (e) {
          console.warn('[S2] 清空目录失败:', e.message);
        }

        // 将用户配置写入临时 JSON 文件
        const configPayload = {};
        if (config.anchorLat != null) configPayload.anchor_lat = config.anchorLat;
        if (config.anchorLon != null) configPayload.anchor_lon = config.anchorLon;
        if (config.anchorAlt != null) configPayload.anchor_alt = config.anchorAlt;
        if (config.numUavs) configPayload.num_uavs = config.numUavs;
        if (config.searchRadius) configPayload.search_radius = config.searchRadius;
        if (config.altitude != null) configPayload.altitude = config.altitude;
        if (config.detectionRange) configPayload.detection_range = config.detectionRange;
        if (config.uavSpeed) configPayload.uav_speed = config.uavSpeed;
        if (config.durationMs) configPayload.duration_ms = config.durationMs;

        await writeFile(S2_CONFIG_FILE, JSON.stringify(configPayload, null, 2), 'utf-8');

        const pyCmd = process.platform === 'win32' ? 'python' : 'python3';
        const proc = spawn(pyCmd, [S2_SCRIPT], {
          cwd: S2_DIR,
          shell: false,
          env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            PYTHONUTF8: '1',
            S2_CONFIG_FILE: S2_CONFIG_FILE
          }
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => { stdout += data.toString(); });
        proc.stderr.on('data', (data) => { stderr += data.toString(); });

        proc.on('close', (code) => {
          uavGenStatus.running = false;
          uavGenStatus.endTime = Date.now();
          if (code !== 0) {
            uavGenStatus.error = stderr || stdout || `exit code ${code}`;
          }
          console.log(`[S2] 无人机轨迹生成完成, exit=${code}, stdout=${stdout.slice(0,200)}`);
        });

        proc.on('error', (err) => {
          uavGenStatus.running = false;
          uavGenStatus.endTime = Date.now();
          uavGenStatus.error = err.message;
        });

        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ok: true, message: '无人机轨迹生成任务已启动' }));
      });

      // GET /api/generate-uav-status — 查询无人机生成任务状态
      server.middlewares.use('/api/generate-uav-status', (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(uavGenStatus));
      });

      // GET /api/list-trace-options — 返回可用的 sat_trace_n 和 uav_trace_n 选项
      server.middlewares.use('/api/list-trace-options', (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        try {
          const satOpts = [];
          const uavOpts = [];
          if (existsSync(S3_TRACES_DIR)) {
            const entries = readdirSync(S3_TRACES_DIR);
            for (const d of entries) {
              const full = join(S3_TRACES_DIR, d);
              if (!statSync(full).isDirectory()) continue;
              if (d.startsWith('sat_trace_')) {
                const n = d.replace('sat_trace_', '');
                if (/^\d+$/.test(n)) satOpts.push(parseInt(n));
              } else if (d.startsWith('uav_trace_')) {
                const n = d.replace('uav_trace_', '');
                if (/^\d+$/.test(n)) uavOpts.push(parseInt(n));
              }
            }
          }
          satOpts.sort((a,b) => a-b);
          uavOpts.sort((a,b) => a-b);
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ sat: satOpts, uav: uavOpts }));
        } catch (e) {
          res.statusCode = 500; res.end(JSON.stringify({ error: e.message }));
        }
      });

      // POST /api/generate-s3 — 启动 S3 模块生成拓扑链路
      server.middlewares.use('/api/generate-s3', async (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }

        if (s3GenStatus.running) {
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, message: '已有生成任务正在运行中' }));
          return;
        }

        let config = {};
        try {
          const chunks = [];
          for await (const chunk of req) { chunks.push(chunk); }
          const body = Buffer.concat(chunks).toString();
          if (body) config = JSON.parse(body);
        } catch (e) { /* 使用默认值 */ }

        if (!config.satN || !config.uavN) {
          res.statusCode = 400;
          res.end(JSON.stringify({ ok: false, message: '请选择卫星和无人机轨迹数量' }));
          return;
        }

        s3GenStatus = { running: true, startTime: Date.now(), endTime: null, error: null };

        // 清空 vis 前端 topology_links 目录中的旧文件
        try {
          if (existsSync(TOPO_DIR)) {
            const files = await readdir(TOPO_DIR);
            await Promise.all(files.map(f => unlink(join(TOPO_DIR, f))));
            console.log(`[S3] 已清空 vis topology_links 目录 (${files.length} 个文件)`);
          }
        } catch (e) {
          console.warn('[S3] 清空目录失败:', e.message);
        }

        const configPayload = { sat_n: config.satN, uav_n: config.uavN };
        if (config.maxSteps) configPayload.max_steps = config.maxSteps;
        await writeFile(S3_CONFIG_FILE, JSON.stringify(configPayload, null, 2), 'utf-8');

        const pyCmd = process.platform === 'win32' ? 'python' : 'python3';
        const proc = spawn(pyCmd, [S3_SCRIPT], {
          cwd: S3_DIR,
          shell: false,
          env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            PYTHONUTF8: '1',
            S3_CONFIG_FILE: S3_CONFIG_FILE
          }
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => { stdout += data.toString(); });
        proc.stderr.on('data', (data) => { stderr += data.toString(); });

        proc.on('close', (code) => {
          s3GenStatus.running = false;
          s3GenStatus.endTime = Date.now();
          if (code !== 0) {
            s3GenStatus.error = stderr || stdout || `exit code ${code}`;
          }
          console.log(`[S3] 拓扑生成完成, exit=${code}, stdout=${stdout.slice(0,200)}`);
        });

        proc.on('error', (err) => {
          s3GenStatus.running = false;
          s3GenStatus.endTime = Date.now();
          s3GenStatus.error = err.message;
        });

        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ok: true, message: 'S3 拓扑生成任务已启动' }));
      });

      // GET /api/generate-s3-status — 查询 S3 生成任务状态
      server.middlewares.use('/api/generate-s3-status', (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(s3GenStatus));
      });

      // GET /api/list-s4-options — 返回可用的 S3 output (sat_n, uav_n) 组合
      server.middlewares.use('/api/list-s4-options', (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        try {
          const opts = [];
          if (existsSync(S3_OUTPUTS_DIR)) {
            const entries = readdirSync(S3_OUTPUTS_DIR);
            for (const d of entries) {
              const full = join(S3_OUTPUTS_DIR, d);
              if (!statSync(full).isDirectory() || !d.startsWith('output_')) continue;
              // 匹配 output_{sat_n}_{uav_n} 格式
              const m = d.match(/^output_(\d+)_(\d+)$/);
              if (m) opts.push({ satN: parseInt(m[1]), uavN: parseInt(m[2]) });
            }
          }
          opts.sort((a,b) => a.satN - b.satN || a.uavN - b.uavN);
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(opts));
        } catch (e) {
          res.statusCode = 500; res.end(JSON.stringify({ error: e.message }));
        }
      });

      // POST /api/generate-s4 — 启动 S4 模块生成 networks.csv
      server.middlewares.use('/api/generate-s4', async (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }

        if (s4GenStatus.running) {
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, message: '已有生成任务正在运行中' }));
          return;
        }

        let config = {};
        try {
          const chunks = [];
          for await (const chunk of req) { chunks.push(chunk); }
          const body = Buffer.concat(chunks).toString();
          if (body) config = JSON.parse(body);
        } catch (e) { /* 使用默认值 */ }

        if (!config.satN || !config.uavN) {
          res.statusCode = 400;
          res.end(JSON.stringify({ ok: false, message: '请选择卫星和无人机轨迹数量' }));
          return;
        }

        s4GenStatus = { running: true, startTime: Date.now(), endTime: null, error: null };

        // 清空 vis 前端 networks 目录中的旧文件
        try {
          if (existsSync(NETWORKS_DIR)) {
            const files = await readdir(NETWORKS_DIR);
            await Promise.all(files.map(f => unlink(join(NETWORKS_DIR, f))));
            console.log(`[S4] 已清空 vis networks 目录 (${files.length} 个文件)`);
          }
        } catch (e) {
          console.warn('[S4] 清空目录失败:', e.message);
        }

        const configPayload = { sat_n: config.satN, uav_n: config.uavN, mode: config.mode || 'b' };
        await writeFile(S4_CONFIG_FILE, JSON.stringify(configPayload, null, 2), 'utf-8');

        const pyCmd = process.platform === 'win32' ? 'python' : 'python3';
        const proc = spawn(pyCmd, [S4_SCRIPT], {
          cwd: S4_DIR,
          shell: false,
          env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            PYTHONUTF8: '1',
            S4_CONFIG_FILE: S4_CONFIG_FILE
          }
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => { stdout += data.toString(); });
        proc.stderr.on('data', (data) => { stderr += data.toString(); });

        proc.on('close', (code) => {
          s4GenStatus.running = false;
          s4GenStatus.endTime = Date.now();
          if (code !== 0) {
            s4GenStatus.error = stderr || stdout || `exit code ${code}`;
          }
          console.log(`[S4] networks 生成完成, exit=${code}, stdout=${stdout.slice(0,200)}`);
        });

        proc.on('error', (err) => {
          s4GenStatus.running = false;
          s4GenStatus.endTime = Date.now();
          s4GenStatus.error = err.message;
        });

        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ok: true, message: 'S4 networks 生成任务已启动' }));
      });

      // GET /api/generate-s4-status — 查询 S4 生成任务状态
      server.middlewares.use('/api/generate-s4-status', (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(s4GenStatus));
      });
    }
  };
}

export default defineConfig({
  plugins: [cesium(), trajectoryApiPlugin()],
  publicDir: 'public',
  root: './',
  server: {
    host: true,
    fs: {
      allow: ['.']
    }
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        earth: resolve(__dirname, 'earth.html'),
        import: resolve(__dirname, 'import.html'),
      }
    }
  }
});
