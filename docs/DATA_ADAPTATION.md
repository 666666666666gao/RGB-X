# 已观察到的数据差异

服务器数据源：`/data/wangwj/dataset`，全部按只读方式使用。
目录数：LasHeR train/test = 979/245，VisEvent train/test = 500/320。
DepthTrack 的 `Dep_train` 有 151 个目录，另有 `toy07_indoor_320` 位于其旁边；test 为 50 个目录。

XTrack 固定版本 `8a606f00c5e98b5393254f6ee1891447b662407f` 的作者划分是
LasHeR 882 train + 97 val，DepthTrack 146 train + 6 val，VisEvent 500 train。
这些是作者代码实际划分，不能写成使用全部 979/150 条训练序列。

`toy07_indoor_320` 出现在作者 train 列表里。项目的 DepthTrack train 目录
逐序列软链接到原目录，并链接这个旁置序列；没有丢弃作者列表里的序列。

作者 LasHeR 加载器预期 `000000.jpg`，现有数据存在 `v000.jpg/i000.jpg` 和其他名称。
统一按照每个序列 visible/infrared 中 JPG 路径的排序读取，保持帧索引对应原注释。
只修改作者 `lasher.py` 的两处路径选择并增加 glob 导入，没有增加猜测式命名分支。
实际图像、标注及作者划分文件不修改。

`pandas==1.5.3` 保留作者所用 `read_csv(..., squeeze=True)` 接口；无需改数据解析代码。
PyTorch/torchvision 使用作者版本 1.13.1/0.14.1，Python 使用 3.8。
Python 3.8 与作者 README 的 3.7 不同，此变化必须保留在环境记录中。

初次审计保留在 `reports/data_inventory_initial.json`。
适配后的结果写入 `reports/data_inventory.json`，实际前向和反向另见 preflight 报告。

全量帧布局审计按 CSV 解析器跳过空行统计标注。`boyouttrees` 和 `girlbikeinlight`
标注文件末尾有空行，不是缺失图像。`bowblkboy1-quezhen` 与 `orange` 的两传感器
使用不同编号体系，因此按原数据排序后的序号配对，不要求文件名数字相等。
帧审计只验证数量与排序，不能据此宣称独立验证了传感器时间同步。

真实数据加载检查发现，作者 `base_functions.py` 创建 validation processing 时
引用了不存在的 `BATProcessing`。该固定版本只有 `XTrackProcessing`，所以将这一处
引用改为 `XTrackProcessing`；训练 processing、本体、损失、冻结范围保持作者定义。
