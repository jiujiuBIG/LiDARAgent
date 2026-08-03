#!/bin/bash

# LiDAR Agent 完整启动脚本
# 启动所有服务：LLM + 后端 + 前端

echo "🌲 LiDAR Agent - 点云智能分析系统"
echo "========================================"
echo ""

# 设置项目根目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ===========================
# 1. 启动 LLM 服务
# ===========================
echo -e "${YELLOW}[1/3]${NC} 启动 LLM 服务 (vLLM)..."
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo -e "      ${GREEN}LLM 服务已在运行${NC}"
else
    nohup ./start_llm.sh > llm_server.log 2>&1 &
    echo -e "      LLM 服务已启动 (PID: $!)"
    echo -e "      等待模型加载..."
    # 等待服务就绪
    for i in {1..30}; do
        if curl -s --max-time 3 http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo -e "      ${GREEN}LLM 服务就绪${NC}"
            break
        fi
        sleep 2
    done
fi

# ===========================
# 2. 启动后端服务
# ===========================
echo -e "${YELLOW}[2/3]${NC} 启动后端服务 (FastAPI :8001)..."

# 确保使用 LLM 环境
export PATH="/home/may/software/anaconda/install/envs/LLM/bin:$PATH"

if pgrep -f "python.*main.py" > /dev/null; then
    echo -e "      ${GREEN}后端服务已在运行${NC}"
else
    cd "$PROJECT_DIR/backend"
    nohup python main.py > ../backend_server.log 2>&1 &
    BACKEND_PID=$!
    echo -e "      后端服务已启动 (PID: $BACKEND_PID)"
    cd "$PROJECT_DIR"
    # 等待后端就绪
    for i in {1..15}; do
        if curl -s --max-time 3 http://localhost:8001/health > /dev/null 2>&1; then
            echo -e "      ${GREEN}后端服务就绪${NC}"
            break
        fi
        sleep 1
    done
fi

# ===========================
# 3. 启动前端服务
# ===========================
echo -e "${YELLOW}[3/3]${NC} 启动前端服务 (Vite :5173)..."

# 加载 nvm 环境
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 20 > /dev/null 2>&1

if pgrep -f "vite" > /dev/null; then
    echo -e "      ${GREEN}前端服务已在运行${NC}"
else
    cd "$PROJECT_DIR/frontend"
    nohup npm run dev > ../frontend_server.log 2>&1 &
    echo -e "      前端服务已启动 (PID: $!)"
    cd "$PROJECT_DIR"
    # 等待前端就绪
    for i in {1..20}; do
        if curl -s --max-time 3 http://localhost:5173 > /dev/null 2>&1; then
            echo -e "      ${GREEN}前端服务就绪${NC}"
            break
        fi
        sleep 1
    done
fi

# ===========================
# 完成
# ===========================
echo ""
echo "========================================"
echo -e "${GREEN}✅ 所有服务已启动!${NC}"
echo "========================================"
echo ""
echo "📍 访问地址:"
echo "   🌐 前端页面: http://localhost:5173"
echo "   📡 API 文档: http://localhost:8001/docs"
echo "   🤖 LLM 状态: http://localhost:8000/v1/models"
echo ""
echo "📝 日志文件:"
echo "   - LLM 服务:    llm_server.log"
echo "   - 后端服务:    backend_server.log"
echo "   - 前端服务:    frontend_server.log"
echo ""
echo "🛑 停止所有服务:"
echo "   pkill -f 'vllm.entrypoints'"
echo "   pkill -f 'python.*main.py'"
echo "   pkill -f 'vite'"
echo "========================================"
