# 远端 GPU 服务器连接与环境说明

> 本文档整理自 `E:\ESRI_World` 中的项目文档、本地 `~/.ssh/config`，以及 2026-11-24 对远端主机的实机核查与 SegmentAnyTree 环境固化记录。

---

## 1. SSH 连接信息

| 项目 | 值 |
|---|---|
| 远端地址 | `100.67.193.74`（端口 `22`） |
| 用户名 | `admin` |
| 主机名 | `DESKTOP-T14A8QV` |
| SSH 目标 | `admin@100.67.193.74` |
| SSH 私钥路径 | `~/.ssh/id_ed25519_remote3070` |

本地 `~/.ssh/config` 中已配置：

```sshconfig
Host 100.67.193.74
  HostName 100.67.193.74
  User admin
  IdentityFile ~/.ssh/id_ed25519_remote3070
```

### 常用连接命令

```powershell
# 直接使用 ~/.ssh/config 中的配置连接
ssh admin@100.67.193.74

# 或显式指定密钥与自动化参数
ssh -o BatchMode=yes -o ConnectTimeout=10 admin@100.67.193.74 hostname
```

* 自动化脚本调用 SSH / SCP 时，建议保留参数：`BatchMode=yes`、`ConnectTimeout=10`、`StrictHostKeyChecking`、`UpdateHostKeys`。
* 远端默认 SSH 登录进入 Windows 环境；执行复杂 PowerShell 命令时注意单引号转义或使用 `powershell -EncodedCommand`。

---

## 2. 远端主机环境（2026-11-24 实机核查）

| 项目 | 实测值 |
|---|---|
| 宿主机系统 | Windows 11 专业版（`10.0.26200.9974`） |
| GPU | `NVIDIA GeForce RTX 3070 Ti`（`8192 MiB` 显存，驱动 `610.88`） |
| TreeLearn 实验环境（Windows 原生） | `E:\LiDARAgent_deep_trees_20261123`（`.venv\Scripts\python.exe`，Python 3.11.9，PyTorch `2.0.0+cu118`） |
| 历史其他项目目录 | `C:\yolo`（`C:/yolo/.venv/Scripts/python.exe`） |
| 磁盘空间（核查时） | `C:\` 剩余约 `310 GiB` |

---

## 3. SegmentAnyTree 运行环境（WSL 2 + Docker，已固化）

### 3.1 WSL 2 与 Docker 配置

| 项目 | 实测值 |
|---|---|
| WSL 发行版 | `Ubuntu-24.04`（WSL 2，内核 `6.18.58.1-microsoft-standard-WSL2`） |
| `/etc/wsl.conf` | `[boot]` / `systemd=true` |
| 默认 WSL 用户 | `lidar`（`uid=1000`，已加入 `docker` 组，可免 `sudo` 直接调用 Docker） |
| Docker Engine | `29.10.0`（WSL 2 内原生安装，已注册 `nvidia` runtime，`nvidia-ctk` 可用） |
| 原始官方镜像 | `maciekwielgosz/segment-any-tree:latest`（ID `e6c6af9780b9`） |
| **固化修复镜像（推荐直接使用）** | **`segment-any-tree:fixed`**（ID `5ade63fee49d`，Dockerfile 位于 `/home/lidar/sat_docker/Dockerfile`） |
| 镜像内置权重 | `/home/nibio/mutable-outside-world/model_file/PointGroup-PAPER.pt`（665 MB） |
| 推荐 WSL 工作目录 | `/home/lidar/sat_work/`（位于 WSL 原生 ext4 文件系统，I/O 性能优于 `/mnt/c`） |

### 3.2 为什么使用 `segment-any-tree:fixed`

官方 `maciekwielgosz/segment-any-tree:latest` 镜像在构建过程中存在包降级残留，导致 4 个旧目录遮蔽同名 `.py`/`.so` 模块并触发 `ImportError`：
* `/usr/local/lib/python3.8/dist-packages/pandas/core/strings`
* `/usr/local/lib/python3.8/dist-packages/pandas/io/parsers`
* `/usr/local/lib/python3.8/dist-packages/pandas/_testing`
* `/usr/local/lib/python3.8/dist-packages/sklearn/metrics/_pairwise_distances_reduction`

`segment-any-tree:fixed` 在不改动官方镜像的基础上仅清理了以上 4 个残留目录，实测验证：
`pandas: 1.1.5 | sklearn: 1.1.3 | torch: 1.9.0+cu111 | cuda: True | gpu: NVIDIA GeForce RTX 3070 Ti`。

### 3.3 标准调用方式（端到端推理）

```bash
# 在远端 WSL (Ubuntu-24.04) 中执行，默认用户 lidar 即可运行：
# 假设输入 .las/.laz 位于 /home/lidar/sat_work/<run>/in，输出目录为 /home/lidar/sat_work/<run>/out
docker run --rm --gpus all \
  --entrypoint bash \
  -e OMP_NUM_THREADS=8 \
  -v /home/lidar/sat_work/<run>/in:/work/in:ro \
  -v /home/lidar/sat_work/<run>/out:/work/out \
  segment-any-tree:fixed \
  run_inference.sh /work/in /work/out true
```

* 推理结果默认输出在 `/home/lidar/sat_work/<run>/out/final_results/*.laz`。
* 已通过 `stage_c04_annotated`（332,594 点）端到端实测：耗时约 `52s`，峰值显存 `1,825 MiB`，原生保留 `source_index` 与 `PredInstance` 字段，可直接对接 `experiments/deep_tree_segmentation/deep_trees.py evaluate --model segmentanytree` 评分。

---

## 4. PKU 多卡 GPU 服务器（`pku-gpu`，vLLM 大模型托管）

> 2026-11-25 实机核查并完成 `ProxyJump` 免密配置与 `forestmind`（32B + LoRA）4 卡 vLLM 部署验真。

### 4.1 SSH 跳板连接信息

| 项目 | 值 |
|---|---|
| 远端地址 | `162.105.149.243`（端口 `22`，校内网段，需经 `100.67.193.74` 跳板） |
| 用户名 | `cjn` |
| 主机名 / 系统 | `admin01-NF5468M6` / Ubuntu 22.04.5 LTS |
| SSH 别名 | `pku-gpu`（或直接 `162.105.149.243`） |
| 跳板机 | `100.67.193.74`（`ProxyJump 100.67.193.74`） |
| SSH 私钥路径 | `~/.ssh/id_ed25519_pku_gpu` |

本地 `~/.ssh/config` 中已配置：

```sshconfig
Host pku-gpu 162.105.149.243
  HostName 162.105.149.243
  User cjn
  ProxyJump 100.67.193.74
  IdentityFile ~/.ssh/id_ed25519_pku_gpu
```

### 4.2 硬件与复用资源

| 项目 | 实测值 |
|---|---|
| GPU | `6 × NVIDIA GeForce RTX 3090`（每张 `24576 MiB`，驱动 `580.82.09`） |
| 内存 | `1.0 TiB`（首次冷读 NFS 权重约 25–30 分钟，进入系统页缓存后重启很快） |
| 本地根分区 / NFS | `/` 剩余约 `106 GiB`；`/mnt/volume4plus` 剩余 `54 TiB`；`/mnt/Huxiaomei2` 剩余 `8.1 TiB` |
| 复用 Conda 环境 | `/home/may/software/anaconda/install/envs/lidaragent`（Python `3.10.20`，PyTorch `2.8.0+cu128`，vLLM `0.11.0`） |
| 复用底座模型 | `/mnt/Huxiaomei2/May/Huxiaomei/Graduation/H/LLMs/code/LLM_Finetune/model/DeepSeek-R1-Distill-Qwen-32B` |
| 复用 LoRA 与模板 | `/mnt/Huxiaomei2/May/Huxiaomei/mcp_llm_new/outputs/auto_lora_sft_1219/checkpoint-100` |
| 启动脚本位置 | `~/lidaragent_vllm/run_forestmind.sh`（封装 `start_vllm_server.sh`，日志在 `~/lidaragent_vllm/logs/`） |

### 4.3 启动服务与本机隧道

```powershell
# 1. 在远端后台启动 4 卡 vLLM（GPU 0,1,2,3，监听远端 127.0.0.1:8000）
ssh pku-gpu 'nohup setsid bash ~/lidaragent_vllm/run_forestmind.sh > ~/lidaragent_vllm/logs/nohup_launch.log 2>&1 < /dev/null &'

# 2. 在本机建立 SSH 端口转发（将远端 8000 映射至本机 127.0.0.1:8000）
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -N -L 127.0.0.1:8000:127.0.0.1:8000 pku-gpu

# 3. 验证本机接口
curl http://127.0.0.1:8000/v1/models
```

### 4.4 如何停止服务（以及具体停止哪部分）

停止分为 **远端大模型服务**（必须停，释放 4 张 3090 显存）和 **本机 SSH 隧道**（可选，释放本机 `8000` 端口）两部分：

1. **远端停止的是什么**：仅终止当前用户 `cjn` 运行的 `start_vllm_server.sh`、`vllm.entrypoints.openai.api_server` 及 4 个张量并行 Worker 进程，释放 **GPU 0–3 显存（约 21.5 GB × 4）** 与远端 `8000` 端口；**不会**删除任何模型权重、`/home/may` 环境或 `~/lidaragent_vllm/` 脚本，也**不会**影响其他用户的进程。
   ```powershell
   # 优雅停止（发送 SIGTERM，仅限 cjn 用户的 vLLM 进程）
   ssh pku-gpu 'pkill -u cjn -f "vllm.entrypoints.openai.api_server|VLLM::|start_vllm_server.sh" || true; sleep 3; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader'

   # 若 10 秒后仍有残留显存，再强制结束（仅限 cjn 用户）
   ssh pku-gpu 'pkill -9 -u cjn -f "vllm.entrypoints.openai.api_server|VLLM::|start_vllm_server.sh" || true'
   ```
2. **本机停止的是什么**：仅关闭本机监听 `127.0.0.1:8000` 的 SSH 端口转发进程（若在前台运行直接 `Ctrl+C`；若在后台运行可按端口查找并结束）：
   ```powershell
   Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
   ```
