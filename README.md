# RGB-X 联合跟踪研究

2028 项目：`/data/gb/rgbx-risk`；Conda：`/data/gb/conda/envs/rgbx-risk`。

当前按用户新要求训练删减版 CADTrack：全部参数随机初始化、不加载 XTrack 或 CADTrack 权重，一份模型联合 RGB-T／RGB-D／RGB-Event。仅用 GPU0／GPU1；GPU0 已实测设置 250W、75℃目标温度。协议：`configs/cad_rgbx_b0_s2026.json`，先运行普通联合 AdamW B0，再进行同起点的主动共同方向配对比较。真实双卡两步预检和 CPU 保存状态检查通过；正式训练已启动，完整三集成绩尚未取得。

旧 XTrack v1、distance0、C 的完整负结果和用户停止后的 D 保存记录均保留，不恢复旧队列，不声称已超过 baseline。

唯一交接文档：[研究与实验完整交接](docs/RGB-X_研究与实验完整交接_2026-10-02.md)。项目、桌面、2028 和 GitHub 维护字节一致副本；当前目标见 `configs/current_goal.json`。不上传数据、权重或 Conda 环境。
