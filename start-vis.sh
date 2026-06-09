#!/usr/bin/env bash
set -e

echo "========================================"
echo "  Star CDN Visual Frontend Launcher"
echo "========================================"
echo ""

cd "$(dirname "$0")/src/vis"

if [ ! -d "node_modules" ]; then
    echo "[INFO] Dependencies not found. Running npm install..."
    echo ""
    npm install
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] npm install failed. Please check Node.js installation."
        exit 1
    fi
    echo ""
    echo "[OK] Dependencies installed."
    echo ""
fi

echo "[START] Launching Vite dev server..."
echo ""
echo "Browser will open automatically once ready."
echo "Press Ctrl+C to stop."
echo ""

npx vite --open --host
