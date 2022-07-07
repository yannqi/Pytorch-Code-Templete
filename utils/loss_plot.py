import numpy as np
import matplotlib.pyplot as plt
train_loss = []
val_loss = []
# 根据Log文件进行画图
for line in open("log/ML-GCN.log","r",encoding='UTF-8'):

    train_loss_data = line.split('Meanloss: ')[-1]
    if train_loss_data[0:4] != str(2022):
        train_loss.append(float(train_loss_data))
    val_loss_data = line.split('Val loss: ')[-1]
    if val_loss_data[0:4] != str(2022):
        val_loss.append(float(val_loss_data))



# 根据CSV文件画图

import pandas as pd
train_loss = []
df = pd.read_csv('output/plot_data/TextCNN.csv')
for i in df['train_acc']:
    train_loss.append(i)



# 创建一个绘图窗口
plt.figure()
 
epochs = range(len(train_loss))
 
# plt.plot(epochs, acc, 'bo', label='Training acc') # 'bo'为画蓝色圆点，不连线
# plt.plot(epochs, val_acc, 'b', label='Validation acc') 
# plt.title('Training and validation accuracy')
# plt.legend() # 绘制图例，默认在右上角
 
plt.figure()
 
plt.plot(epochs, train_loss,marker = "o",markersize=2, label='Training loss')
plt.plot(epochs, val_loss,marker = "o",markersize=2, label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
 
plt.show()
plt.savefig('output/plot_data/loss.png')