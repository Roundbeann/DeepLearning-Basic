# # import matplotlib.pyplot as plt
# # def fun3(x):
# #     return 5 * x**3 - 2* x**2 + 3*x  - 4
# #
# # x = []
# # y = []
# #
# # i = 4
# # a=[]
# # while (i<6):
# #     i+=0.10
# #     a.append(i)
# # print("a=",a)
# # a = [4.1, 4.199999999999999, 4.299999999999999, 4.399999999999999, 4.499999999999998, 4.599999999999998, 4.6999999999999975, 4.799999999999997, 4.899999999999997, 4.9999999999999964, 5.099999999999996, 5.199999999999996, 5.299999999999995, 5.399999999999995, 5.499999999999995, 5.599999999999994, 5.699999999999994, 5.799999999999994, 5.899999999999993, 5.999999999999993, 6.0999999999999925]
# #
# # for i in a:
# #     x.append(i)
# #     y.append(fun3(i))
# # plt.plot(x,y)
# # plt.show()
# # print("x=",x,"\ny=",y)
# # x = [[1,2,3],[-1,2,3],[-1,5,2],[2,-3,-4],[-2,-3,5],[2,3,-6]]
# # def fun (l):
# #     x = l[0]
# #     y = l[1]
# #     z = l[2]
# #     return 3*x+4*y-5*z+3.5
# # for i in x:
# #     print(fun(i))
# #
# #
#
#
# import numpy as np # 用来处理数据
# import matplotlib.pyplot as plt
#
# X = np.array([[3,2],
# [4,3],
# [5,3],
# [7,4],
# [8,7],
# [9,7],
# [12,11],
# [3,1],
# [4,1],
# [5,2],
# [7,3],
# [8,3],
# [9,4],
# [12,5]])
#
# x = X[:,0]
# y = X[:,1]
# z = np.array([[1],[1],[1],[1],[1],[1],[1],[-1],[-1],[-1],[-1],[-1],[-1],[-1]])
#
# ax = plt.subplot(projection = '3d') # 创建一个三维的绘图工程
# ax.set_title('3d_image_show') # 设置本图名称
# ax.scatter(x, y, z, c = 'r') # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
#
# ax.set_xlabel('X') # 设置x坐标轴
# ax.set_ylabel('Y') # 设置y坐标轴
# ax.set_zlabel('Z') # 设置z坐标轴
#
# plt.show()

# a = [1,2,3,4,5,6,7]
# print(a[0:8])

import torch.nn as nn
import torch
embedding = nn.Embedding(10,20)
data1 = torch.tensor([0,1,5,9])
data2 = torch.tensor([[0,1,9],[0,0,0,0,0]])
data3 = torch.tensor([[[0,1,5,6,9],[0,0,0,0,0]],[[0,1,5],[0,0,0,0,0]],[[0],[0,0,0,0,0]]])
embed1 = embedding(data1)
embed2 = embedding(data2)
embed3 = embedding(data3)
out = torch.tensor([25])
# 超出num_enbeddings 也就是超出字典的长度的字，得不到对应的embedding
# 每个字用embedding_dim维的向量来表示
outembed = embedding(out)
print()