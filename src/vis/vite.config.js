import { defineConfig } from 'vite';
import cesium from 'vite-plugin-cesium';
import { resolve, join } from 'path';
import { exec, spawn } from 'child_process';
import { writeFile, mkdir, readdir, unlink } from 'fs/promises';
import { existsSync } from 'fs';

const DATA_DIR = resolve(__dirname, 'public/data');
const SAT_TRACE_DIR = join(DATA_DIR, 'sat_trace');
const UAV_TRACE_DIR = join(DATA_DIR, 'uav_trace');
const TOPO_DIR = join(DATA_DIR, 'topology_links');
const NETWORKS_DIR = join(DATA_DIR, 'networks');
const S1_SCRIPT = resolve(__dirname, '..', 'S1', 'main_s1.py');
const S1_DIR = resolve(__dirname, '..', 'S1');

// 生成任务状态（内存中）
let genStatus = { running: false, startTime: null, endTime: null, error: null };

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

        if (genStatus.running) {
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, message: '已有生成任务正在运行中' }));
          return;
        }

        genStatus = { running: true, startTime: Date.now(), endTime: null, error: null };

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

        const pyCmd = process.platform === 'win32' ? 'python' : 'python3';
        const proc = spawn(pyCmd, [S1_SCRIPT], {
          cwd: S1_DIR,
          shell: false,
          env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' }
        });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => { stdout += data.toString(); });
        proc.stderr.on('data', (data) => { stderr += data.toString(); });

        proc.on('close', (code) => {
          genStatus.running = false;
          genStatus.endTime = Date.now();
          if (code !== 0) {
            genStatus.error = stderr || `exit code ${code}`;
          }
          console.log(`[S1] 卫星轨迹生成完成, exit=${code}`);
        });

        proc.on('error', (err) => {
          genStatus.running = false;
          genStatus.endTime = Date.now();
          genStatus.error = err.message;
        });

        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ok: true, message: '卫星轨迹生成任务已启动' }));
      });

      // GET /api/generate-status — 查询生成任务状态
      server.middlewares.use('/api/generate-status', (req, res) => {
        if (req.method !== 'GET') { res.statusCode = 405; res.end(); return; }
        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(genStatus));
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
