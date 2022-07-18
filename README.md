# Templete

# 介绍
Pytorch代码模板，用于快速建立一个简单的项目

<<<<<<< HEAD
=======
该模板持续更新中...

该仓库实现GitHub与Gitee同步

#
>>>>>>> 34449d9a1e7b5bfa27c1e285dcec7d31131906ca

代码应包含如下文件夹
- config : 用于存储config文件
- data :  dataloader.py 及 dataset 存储
- log : 实时记录log文件
- utils : 一些其他的函数记录在此处
- image : 存储一些pipline图

# 一些模板规范的思考

在argparse和config的设置中感到迷惑，代码界内没有形成统一的规范。而我本人也未达成一致的规范。
在参考NVIDIA的一些官方代码和YOLO的代码后，发现对可以明面上设置的参数，如batch_size, epoch, learning_rate 这些参数用argparse设置更为方便，后续可用scripts中的sh规范化运行。
而对于一些模型中的参数或者数据集的参数，基本上确定后不再改变的设置参数，则放到config内更为方便。
后续在代码完善的过程中，会对诸如此类的代码模板做出更新。