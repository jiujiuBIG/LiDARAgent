## 环境要求

### 硬件要求

模型运行建议使用以下 GPU 环境：

- GPU：4 张 NVIDIA RTX 3090
- 单卡显存：24 GB
- 总显存：96 GB

`start_vllm_server.sh` 默认使用 4 张 GPU，并设置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
TENSOR_PARALLEL_SIZE=4
```
如果实际 GPU 编号不同，可在启动前通过环境变量调整。

### 软件要求
建议环境如下：
- Linux 或支持 Bash 的服务器环境
- NVIDIA Driver 与 CUDA 运行环境
- Conda
- Python 3.10
- Node.js 与 npm
- vLLM
- pybind11 以及项目依赖中的 Python 库


## 安装步骤

### 1. 克隆或进入项目目录
```bash
cd LiDARAgent
```

### 2. 创建并激活 Python 环境
```bash
conda create -n lidaragent python=3.10 -y
conda activate lidaragent
```

### 3. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

## 安装项目库文件
```bash
find mcp_llm_web/backend/tools/lib -name setup.py -print -execdir python setup.py install \;
```
安装本地库前，请确保当前 Python 环境已经安装编译相关依赖，并且系统具备可用的 C/C++ 编译器。


## 启动大语言模型服务
```bash
bash start_vllm_server.sh
```

## 启动后端服务
```bash
cd mcp_llm_web/backend
python main.py
```

## 启动前端服务
```bash
cd mcp_llm_web/frontend
npm run dev
```
终端会输出访问地址，通常为：http://localhost:5173，在浏览器中打开该地址即可进入 Web 应用界面。

