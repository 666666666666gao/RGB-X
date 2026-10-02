# RGB-X 定位风险驱动的联合训练

2028 服务器项目：`/data/gb/rgbx-risk`。
Conda 环境：`/data/gb/conda/envs/rgbx-risk`。

研究范围是 RGB-T、RGB-D、RGB-Event 联合训练，同一份模型权重支持三类任务。
主研发基线为官方 XTrack-B；第二结构为 SUTrack-B224，B384 放在方法有效后。
训练数据复用服务器已有 LasHeR、DepthTrack、VisEvent，不复制原始图像。

当前工作只建立项目、环境、路径和训练前检查。方法设计、优化器对照与正式实验的完成状态见 `docs/EXPERIMENT_PLAN.md` 和 `reports/SETUP_STATUS.md`。
文献指标是作者报告，不是本项目复现结果。

激活环境：

```bash
source /data/liangds/anaconda3/etc/profile.d/conda.sh
conda activate /data/gb/conda/envs/rgbx-risk
cd /data/gb/rgbx-risk
```

官方代码来源：[XTrack](https://github.com/supertyd/XTrack)、[SUTrack](https://github.com/chenxin-dlut/SUTrack)。

`scripts/train_xtrack_adamw.sh` 使用 65 轮作者配方的低磁盘保存版本，只保存最后一轮。
它没有中间恢复点，也不通过测试集选权重；正式对照须统一保存与选模规则。
此入口只训练，不自动评测。当前尚未启动正式训练。
