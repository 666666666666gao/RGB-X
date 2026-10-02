# 2028 项目部署完成（2026-10-02）

项目：`/data/gb/rgbx-risk`。
Conda 环境：`/data/gb/conda/envs/rgbx-risk`，Python 3.8.20。
本机副本：`C:\Users\gb\projects\rgbx-risk`。
PyTorch 1.13.1+cu117、torchvision 0.14.1+cu117、CUDA runtime 11.7。
GPU：三张 RTX 3090；设备发现与 CUDA 矩阵运算通过，三任务模型预检使用 GPU 0。
`pip check` 与 TensorBoard 写入通过。完整版本见 `requirements-lock.txt`、`conda-explicit.txt`。

## 已完成

- 固定 XTrack 作者提交 `8a606f00c5e98b5393254f6ee1891447b662407f`。
- 固定 SUTrack 作者提交 `d65052d1ba3fcf55010e1fb3665ee6616c139a2c`；仅准备第二结构源码，尚未完成其运行适配。
- 六个训练/测试根路径用软链接接入，原始图像不复制、不修改。
- DepthTrack train 做序列链接覆盖旁置的 `toy07_indoor_320`。
- 作者实际 train/val：LasHeR 882/97，DepthTrack 146/6；VisEvent train 500。
- 当前三项 test 数据规模：LasHeR 245、DepthTrack 50、VisEvent 320。
- LasHeR 979 个训练池序列的帧数量与排序审计通过；同步真实性的限制见帧审计说明。
- 复用 OSTrack 初始化权重；SHA256 与作者 Hugging Face LFS OID 相同：
  `8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41`。
- 修正本机路径、LasHeR 文件名读取以及作者入口的不存在 processing 引用，完整 diff 见 `XTrack_local_changes.patch`。
- 保留原作者保存配置，并提供只保存 E65 最终 checkpoint 的低磁盘版本。
- 三模块设计、第二主模块任务自适应机制、必做对照与迁移顺序已写入实验计划。

## 实际 GPU 预检

同一个 XTrack-B 模型使用 T/D/E 各 2 个真实样本，完成三次前向/反向，
平均三个任务梯度并完成一次 AdamW 更新。没有保存训练权重。
242 个作者初始化张量与模型内容逐一完全相同；新增时序位置参数保留作者行为。

| 项目 | 实际结果 |
|---|---:|
| 总参数量 | 98,341,509 |
| 可训练参数量 | 5,821,440 |
| 可训练张量数 | 504 |
| 更新改变的可训练张量数 | 489 |
| 冻结参数 | 梯度为空且版本未改变 |
| 峰值 CUDA 显存 | 1,843,556,864 bytes（约 1.72 GiB） |
| 预检耗时 | 7.81 秒 |

完整结果见 `preflight_xtrack.json`。这些损失、显存和时间属于 batch=2 工程预检，
不是完整训练吞吐、正式跟踪指标或方法增益。

## 尚未完成

正式训练没有启动；PCGrad、CAGrad、AdaTask、三个候选模块和 SUTrack 迁移尚未实现/运行。
五个核心评测没有完成。按用户当前要求，RGBT234 和 VOT-RGBD2022 暂缓，未下载、未评测。
FE108、COESOT 放在后续扩展阶段。训练比较不能将作者公布成绩当成本项目复现成绩。

部署后 `/data` 可用空间约 6.0 GB，环境实际占用约 4.0 GB。
低磁盘配置只保存最后一轮，不提供中间恢复点或中间选模候选。
正式多方法、多种子实验需要统一预算、保存策略与空间安排。

## 使用

```bash
source /data/liangds/anaconda3/etc/profile.d/conda.sh
conda activate /data/gb/conda/envs/rgbx-risk
cd /data/gb/rgbx-risk
export PYTHONPATH=/data/gb/rgbx-risk/third_party/XTrack
# 工程预检；模型只在内存中更新一次
python scripts/preflight_xtrack.py
# 启动单 GPU 原配方 AdamW 联合训练（当前未执行）
bash scripts/train_xtrack_adamw.sh
```

配置脚本 `configure_xtrack.py` 用于全新检出的源码，当前已执行；无需在已配置目录重复运行。
正式任务的监视按 240 秒间隔和吞吐 ETA 安排，不在本次启动训练监视进程。
