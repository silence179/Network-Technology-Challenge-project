# Star CDN Visual Frontend - One-click Launcher (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Star CDN 可视化前端 - 一键启动" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$visDir = Join-Path $PSScriptRoot "src\vis"
Set-Location $visDir

if (-not (Test-Path "node_modules")) {
    Write-Host "[信息] 依赖未安装，正在执行 npm install..." -ForegroundColor Gray
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 依赖安装失败，请检查 Node.js 是否正确安装。" -ForegroundColor Red
        Read-Host "按 Enter 退出"
        exit 1
    }
    Write-Host "[完成] 依赖安装完毕。" -ForegroundColor Green
}

Write-Host "[启动] 正在启动 Vite 开发服务器..." -ForegroundColor Gray
Write-Host ""
Write-Host "浏览器将自动打开，按 Ctrl+C 可停止服务。" -ForegroundColor Gray
Write-Host ""

npm run dev

Read-Host "按 Enter 退出"
