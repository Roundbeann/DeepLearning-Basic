import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def get_imgs(path = "img"):
    # files:img目录下所有的图片（文件）名
    files = os.listdir(path)
    imgs = []

    for file in files :
        # 根据图片路径，读取图片内容
        # img【218 178 3】
        # img[3][0:2]:【[[ 60  75 121], [ 60  75 121]]】
        img = cv2.imread(os.path.join(path,file))
        # 把图片大小设置为 150，150
        img = cv2.resize(img,(150,150))

        #这里正对opencv和caffe图像格式转化说明一下：
        # opencv存储图片使用的是：H×W×Chanel
        # caffe存储图片使用的是： Chanel×H×W 【3 150 150】
        img = img.transpose(2,0,1)
        imgs.append(img)
    return np.array(imgs)


def conv2d(kernel, imgs):
    imgNum,outputChanel,inputChanel,kernelHeight,kernelWeight,imgHeight,imgWeight= (
        imgs.shape[0],kernel.shape[0],kernel.shape[1],kernel.shape[2],kernel.shape[3],imgs.shape[2],imgs.shape[3]
    )
    padding = 0
    stride = 1
    featureMapHeight = int((imgHeight - kernelHeight + 2 * padding) / stride + 1)
    featureMapWeight = int((imgWeight - kernelWeight + 2 * padding) / stride + 1)

    kernelMatrix = kernel.reshape(outputChanel,-1)

    # Implement 1.------------start
    imgMatrixs =[]
    for img in imgs:
        imgMatrix = np.zeros(shape=(kernelMatrix.shape[1], featureMapHeight * featureMapWeight))
        i = 0
        for h in range(featureMapHeight):
            for w in range(featureMapWeight):
                area = img[:,h:h+kernelHeight,w:w+kernelWeight].reshape(-1)
                imgMatrix[:,i] = area
                i = i+1
        imgMatrixs.append(imgMatrix)
    # kernelMatrix 【2 27】
    # imgMatrixs【4 27 21904】
    imgMatrixs = np.array(imgMatrixs)
    # Implement 1.------------stop

    # 利用广播机制
    # Implement 2.
    # imgMatrixs 【4 27 21904】
    # ------------start 对广播机制还不是很熟悉
    # imgMatrixs = np.zeros(shape=(imgNum, kernelMatrix.shape[1], featureMapHeight * featureMapWeight))
    # i = 0
    # for h in range(featureMapHeight):
    #     for w in range(featureMapWeight):
    #         area = imgs[:,:,h:h+kernelHeight,w:w+kernelWeight].reshape(imgNum,-1)
    #         imgMatrixs[:,:,i] = area
    #         i = i+1
    # imgMatrixs = np.array(imgMatrixs)
    # ------------stop
    # Implement 2.



    res = kernelMatrix @ imgMatrixs
    res = res.reshape(res.shape[0],res.shape[1],featureMapHeight,featureMapWeight)

    return res


if __name__ == "__main__":

    kernel = np.array(
    [
        [
            [
                [-1,-2,-3],
                [-1,-2,-3],
                [1,1,1]
            ],
            [
                [3,3,3],
                [-1,-2,-3],
                [1,1,1]
            ],
            [
                [3,3,3],
                [-1,-2,-3],
                [-1,-2,-3]
            ]
        ],
            [
                [
                    [1, -1, 0],
                    [1, -1, 0],
                    [1, -1, 0]
                ],
                [
                    [1, -1, 0],
                    [1, -1, 0],
                    [1, -1, 0]
                ],
                [
                    [1, -1, 0],
                    [1, -1, 0],
                    [1, -1, 0]
                ]
            ]
        ]
    )

    # imgs 【3 3 150 150】
    # 3张图片 每张图片3个通道
    imgs = get_imgs()
    result = conv2d(kernel,imgs)
    for i in result:
        for j in i:
            plt.imshow(j,cmap="gray")
            plt.show()
    print(kernel.shape)



#     对于卷积操作 batch 永远放在数据维度的第一个位置
#     3 张 原始图片（每张图片3个通道）
#     2 个 卷积核（每个卷积核3个通道）
#     2 个 卷积核 对 3 张 原始图片 做卷积运算
#     每张原始图片得到 2 张 特征图
#     3 张 原始图片 得到 6 张 特征图
#     卷积运算的结果形状为 【3 2 * *】 * 为特征图的大小
