# RGB-X 定位风险驱动联合训练

2028项目：`/data/gb/rgbx-risk`；Conda：`/data/gb/conda/envs/rgbx-risk`。

一份联合权重支持RGB-T、RGB-D、RGB-Event，复用现有LasHeR、DepthTrack、VisEvent。

用户最新目标：直接训练完整三个模块，以完整三集指标超过baseline为目标，未达到则诊断并继续优化。原AdamW复现已按用户要求中断：完成29轮与第30轮1650步，没有可评测终轮权重。原始日志及中断报告保留。

当前完整方法源码已实现，独立审阅与三卡真实预检尚待通过，新GPU训练尚未启动，正式跟踪结果仍0/3。使用作者公开XTrack-B联合权重初始化与同版本评测参考，额外微调预算会单独报告；不宣称与作者原训练同预算。

协议：`configs/full_method_v1.json`；当前目标：`configs/current_goal.json`。2028仅使用三卡0/1/2。其他四个测试集暂缓。

唯一交接文档：[研究与实验完整交接](docs/RGB-X_研究与实验完整交接_2026-10-02.md)。项目、桌面与服务器维护同一份字节一致文档，JSON、日志和补丁记录原始证据。

官方来源：[XTrack](https://github.com/supertyd/XTrack)、[公开权重](https://huggingface.co/taryya/XTrack/tree/main)、[SUTrack](https://github.com/chenxin-dlut/SUTrack)。仓库不上传数据、模型权重或Conda环境。
