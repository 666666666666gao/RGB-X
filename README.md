# RGB-X 定位风险驱动联合训练

2028项目：`/data/gb/rgbx-risk`；Conda：`/data/gb/conda/envs/rgbx-risk`。

完整三个训练模块已于2026-10-03 14:55在2028三卡0/1/2启动。RGB-T、RGB-D、RGB-Event共同训练一份权重，使用已有LasHeR、DepthTrack、VisEvent。协议：`configs/full_method_v1.json`，run：`xtrack_full_method_v1_s2026`，15轮公开联合权重上的微调，额外训练预算单独报告。

原AdamW复现按用户要求中断，29轮及第30轮1650步，无可评测终轮权重。完整方法源码审阅和两次真实三卡预检通过，训练仍在进行，正式跟踪评测0/3，尚未证明超过baseline。

完成训练后自动对公开参考和完整方法做现有三集完整评测，七个主指标逐项比较，补齐属性、曲线、逐序列、失败和成本。未全面超过则继续诊断改进。其他四测试集暂缓。

唯一交接文档：[研究与实验完整交接](docs/RGB-X_研究与实验完整交接_2026-10-02.md)，项目、桌面、2028及GitHub维护字节一致副本。当前目标：`configs/current_goal.json`。不上传数据、权重或Conda环境。
