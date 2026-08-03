#!/bin/bash
# LiDAR Agent Web 启动脚本

echo "🌲 LiDAR Agent - 点云智能分析系统"
echo "================================"

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到 Python 环境"
    exit 1
fi

# 启动后端
echo "🚀 启动后端服务 (端口 8001)..."
cd "$(dirname "$0")/backend"
python main.py &
BACKEND_PID=$!

echo "后端 PID: $BACKEND_PID"

# 等待后端启动
sleep 3

# 提示用户
echo ""
echo "✅ 服务已启动!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📡 API 文档: http://localhost:8001/docs"
echo "🌐 前端页面: 直接在浏览器中打开 frontend/index.html"
echo ""
echo "停止服务: kill $BACKEND_PID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
