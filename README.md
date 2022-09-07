# Templete

# 介绍
Pytorch代码模板，用于快速建立一个简单的项目

该模板持续更新中...

该仓库实现GitHub与Gitee同步

代码应包含如下文件夹(无先后顺序)
- config : 用于存储config文件
- data :  用于加载数据集的py文件
- log : 实时记录log文件
- utils : 一些其他的函数记录在此处
- image : 存储一些pipline/loss图
- checkpoints : 存储保存的模型
- scripts : 存储便捷运行代码的Bash文件

# 模板包含功能

## 基础功能
- [x] Vscoded下 Python上下级包导入。见`utils.utils.py` 函数：`add_sys_path(path)`



- [x] Log记录。 `utils/Logger.py`
- [x] 数据集的加载。 `data/CatVsDog.py`
- [ ] 数据集的预处理及增强
- [ ] scripts sh文件
- [x] 数据集config文件管理。
- [ ] Argparse规范化
## 扩展功能

- [x] AMP(Automatic Mixed Precision)，自动混合精度。混合FP16和FP32精度，在不损失模型性能的前提下，提高2x~5x训练速度。
- [ ] 多GPU，DDP训练 
- [x] Markdown参考。`README_TEMPLETE.md`

# 一些模板规范的思考


1. 在argparse和config的设置中感到迷惑，代码界内没有形成统一的规范。而我本人也未达成一致的规范。
在参考NVIDIA的一些官方代码和YOLO的代码后，发现对可以明面上设置的参数，如batch_size, epoch, learning_rate 这些参数用argparse设置更为方便，后续可用scripts中的sh规范化运行。
而对于一些模型中的参数或者数据集的参数，基本上确定后不再改变的设置参数，则放到config内更为方便。后续在代码完善的过程中，会对诸如此类的代码模板做出更新。
    - 后续更新：

