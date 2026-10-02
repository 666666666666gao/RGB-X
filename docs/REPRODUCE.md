# 固定源码与部署说明

本仓库提供实验配置、部署/预检/启动脚本、作者源码补丁和实验记录。`third_party/`、`data/`、`pretrained/`、`outputs/`、`logs/` 不提交到 GitHub。作者源码保留原项目许可证；本项目没有重新授予其许可证。

## 当前服务器：只接续观察

2028 上的 `/data/gb/rgbx-risk` 已配置完成，正式训练正在运行。不要在此目录重新执行下面的源码获取、环境安装、数据配置、预检或正式启动命令。

```bash
cd /data/gb/rgbx-risk
cat reports/xtrack_b_adamw_3gpu_s2026_20261002.json
tail -n 6 outputs/xtrack_b_adamw_3gpu_s2026_20261002/logs/xtrack-rgbx_b_adamw_3gpu.log
tmux list-sessions
df -h /data
```

## 全新部署

现有路径脚本面向 2028 的 `/data/gb/rgbx-risk`。换机器或换路径时，应明确修改数据根目录、初始化权重和环境前缀，再做真实数据预检。

```bash
git clone https://github.com/666666666666gao/RGB-X.git /data/gb/rgbx-risk
cd /data/gb/rgbx-risk
bash scripts/fetch_sources.sh
```

固定作者版本：

| 源码 | 仓库 | 提交 |
|---|---|---|
| XTrack | https://github.com/supertyd/XTrack | `8a606f00c5e98b5393254f6ee1891447b662407f` |
| SUTrack | https://github.com/chenxin-dlut/SUTrack | `d65052d1ba3fcf55010e1fb3665ee6616c139a2c` |

环境安装入口为 `bash scripts/install_environment.sh`，完整实际版本另见 `reports/requirements-lock.txt` 和 `reports/conda-explicit.txt`。本次已验证 Python 3.8.20、PyTorch 1.13.1+cu117、torchvision 0.14.1+cu117；安装后必须确认 `torch.cuda.is_available()` 和三卡 CUDA witness 通过。构建过程遇到的问题写在完整交接文档，不能只把 pip 安装成功当作 GPU 环境通过。

原始数据需要已经位于 `/data/wangwj/dataset`，初始化权重需要位于 `/data/qianmj/BASE/pretrained/OSTrack_ep0300.pth.tar`，其 SHA256 必须为：

```text
8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41
```

在全新数据 facade 与未修改的固定版本 XTrack 上，以下操作执行一次：

```bash
mkdir -p data pretrained outputs logs reports
export PYTHONPATH="$PWD/third_party/XTrack"
/data/gb/conda/envs/rgbx-risk/bin/python scripts/configure_xtrack.py
/data/gb/conda/envs/rgbx-risk/bin/python scripts/prepare_xtrack_3gpu.py
```

`configure_xtrack.py` 创建数据软链接、修复 LasHeR 文件名读取和验证 processing，并生成基础/低磁盘配置。`prepare_xtrack_3gpu.py` 生成三卡配置并将作者现有 `fail_safe` 置为 False。这两个脚本是一次性准备步骤，不是可重复的状态刷新命令。它们实施的四处作者源码变化见 `reports/XTrack_local_changes.patch`；不要再重复应用该补丁。

验证命令与预期结果见 [RUN_20261002.md](RUN_20261002.md)。三卡 smoke 的输出目录必须属于新的一次检查；已存在的 smoke checkpoint 会被作者 `load_latest=True` 读取，不能把重读已有 checkpoint 当作重新执行了两次更新。

只有真实预检通过后，才为一个新的、明确授权的实验 run ID 调用 `scripts/run_xtrack_job.py`。当前正式运行的名字已经占用，不能复用。`scripts/finalize_setup.py` 是历史建项收据脚本，会写入“尚未正式训练”的旧状态，不要在已经启动训练的项目中重跑。
