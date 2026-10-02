# RGB-X 定位风险驱动的联合训练

2028 服务器项目：`/data/gb/rgbx-risk`。
Conda 环境：`/data/gb/conda/envs/rgbx-risk`。

研究范围是 RGB-T、RGB-D、RGB-Event 联合训练，同一份模型权重支持三类任务。
主研发基线为官方 XTrack-B；第二结构为 SUTrack-B224，B384 放在方法有效后。
训练数据复用服务器已有 LasHeR、DepthTrack、VisEvent，不复制原始图像。

截至 2026-10-02 22:13（北京时间），单卡预检、三卡 DDP 预检和划分核查通过；一份联合权重的 XTrack-B 作者 AdamW 基线已完成五轮和首次验证，进入第 6/65 轮。每卡 batch 8、全局 batch 24，种子 2026。尚无正式测试集指标。

部署、数据、参数范围、实测结果、问题和后续计划统一维护在唯一的 [研究与实验交接文档](docs/RGB-X_研究与实验完整交接_2026-10-02.md)。README 只提供入口；JSON、日志和补丁保留作原始证据。方法模块、PCGrad/CAGrad/AdaTask 等对照和 SUTrack 训练尚未实施。文献指标是作者报告，不是本项目复现结果。

激活环境：

```bash
source /data/liangds/anaconda3/etc/profile.d/conda.sh
conda activate /data/gb/conda/envs/rgbx-risk
cd /data/gb/rgbx-risk
```

官方代码来源：[XTrack](https://github.com/supertyd/XTrack)、[SUTrack](https://github.com/chenxin-dlut/SUTrack)。

当前运行使用 `scripts/train_xtrack_3gpu.sh` 和配置 `rgbx_b_adamw_3gpu`，run ID 为 `xtrack_b_adamw_3gpu_s2026_20261002`。`scripts/run_xtrack_job.py` 在 tmux 会话 `rgbx_adamw_20261002` 内每 240 秒记录状态；SSH 断开不会终止它。

65 轮训练只保存最终 checkpoint，没有中间恢复点。不要在正在训练的项目上重跑初始化或启动命令。只读查看：

```bash
cat reports/xtrack_b_adamw_3gpu_s2026_20261002.json
tail -n 6 outputs/xtrack_b_adamw_3gpu_s2026_20261002/logs/xtrack-rgbx_b_adamw_3gpu.log
```

克隆本项目后，在全新目录执行 `bash scripts/fetch_sources.sh` 获取固定提交的作者源码；完整部署步骤见主交接文档第 15 节。作者源码使用其原有许可证，项目保留来源和修改补丁。仓库不包含数据、权重、Conda 环境或训练输出。

RGBT234、VOT-RGBD2022 因磁盘不足暂缓；五个核心测试集仍属于最终计划，FE108、COESOT 随后补充。
