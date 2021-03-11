
# -*- coding:utf-8 -*-
# author : Han
# date : 2021/3/4 21:41
# IDE : PyCharm
# FILE : lbp.py
import numpy as np
import os
import cv2


class LBP(object):
    def __init__(self, threshold, dsize, blockNum, dirName):
        self.dsize = dsize  # 统一尺寸大小
        self.blockNum = blockNum  # 分割块数目
        self.threshold = threshold  # 阈值，暂未使用
        self.imgList, self.label = self.loadImagesList(dirName)
        self.createTable()


    def loadImg(self, fileName, dsize):
        '''
        载入图像，灰度化处理，统一尺寸，直方图均衡化
        :param fileName: 图像文件名
        :param dsize: 统一尺寸大小。元组形式
        :return: 图像矩阵
        '''
        img = cv2.imread(fileName)
        retImg = cv2.resize(img, dsize)
        retImg = cv2.cvtColor(retImg, cv2.COLOR_BGR2GRAY)
        retImg = cv2.equalizeHist(retImg)
        # cv2.imshow('img',retImg)
        # cv2.waitKey()
        return retImg

    def loadImagesList(self, dirName):
        '''
        加载图像矩阵列表
        :param dirName:文件夹路径
        :return: 包含最原始的图像矩阵的列表和标签矩阵
        '''
        imgList = []
        label = []
        for parent, dirnames, filenames in os.walk(dirName):
            # print parent
            # print dirnames
            # print filenames
            for dirname in dirnames:
                for subParent, subDirName, subFilenames in os.walk(parent + '/' + dirname):
                    for filename in subFilenames:
                        img = self.loadImg(subParent + '/' + filename, self.dsize)
                        imgList.append(img)  # 原始图像矩阵不做任何处理，直接加入列表
                        label.append(subParent)
        return imgList, label

    def getHopCounter(self, num):
        '''
        计算二进制序列是否只变化两次
        :param num: 数字
        :return: 01变化次数
        '''
        binNum = bin(num)
        binStr = str(binNum)[2:]
        n = len(binStr)
        if n < 8:
            binStr = "0" * (8 - n) + binStr
        n = len(binStr)
        counter = 0
        for i in range(n):
            if i != n - 1:
                if binStr[i + 1] != binStr[i]:
                    counter += 1
            else:
                if binStr[0] != binStr[i]:
                    counter += 1
        return counter

    def createTable(self):
        '''
        生成均匀对应字典
        :return: 均匀LBP特征对应字典
        '''
        self.table = {}
        temp = 1
        for i in range(256):
            if self.getHopCounter(i) <= 2:
                self.table[i] = temp
                temp += 1
            else:
                self.table[i] = 0
        return self.table

    def getLBPfeature(self, img):
        '''
        计算LBP特征
        :param img:图像矩阵
        :return: LBP特征图
        '''
        m = img.shape[0];
        n = img.shape[1]
        neighbor = [0] * 8
        featureMap = np.mat(np.zeros((m, n)))
        for y in range(1, m - 1):
            for x in range(1, n - 1):
                neighbor[0] = img[y - 1, x - 1]
                neighbor[1] = img[y - 1, x]
                neighbor[2] = img[y - 1, x + 1]
                neighbor[3] = img[y, x + 1]
                neighbor[4] = img[y + 1, x + 1]
                neighbor[5] = img[y + 1, x]
                neighbor[6] = img[y + 1, x - 1]
                neighbor[7] = img[y, x - 1]
                center = img[y, x]
                temp = 0
                for k in range(8):
                    temp += (neighbor[k] >= center) * (1 << k)
                featureMap[y, x] = self.table[temp]
        featureMap = featureMap.astype('uint8')  # 数据类型转换为无符号8位型，如不转换则默认为float64位，影响最终效果
        return featureMap

    def calcHist(self, roi):
        '''
        计算直方图
        :param roi:图像区域
        :return: 直方图矩阵
        '''
        hist = cv2.calcHist([roi], [0], None, [59], [0, 256])  # 第四个参数是直方图的横坐标数目，经过均匀化降维后这里一共有59种像素
        return hist

    def compare(self, sampleImg, testImg):
        '''
        比较函数，这里使用的是欧氏距离排序，也可以使用KNN，在此处更改
        :param sampleImg: 样本图像矩阵
        :param testImg: 测试图像矩阵
        :return: k2值
        '''
        testImg = cv2.resize(testImg, self.dsize)
        testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
        testFeatureMap = self.getLBPfeature(testImg)
        sampleFeatureMap = self.getLBPfeature(sampleImg)
        # 计算步长，分割整个图像为小块
        ystep = self.dsize[0] // self.blockNum
        xstep = self.dsize[1] // self.blockNum
        k2 = 0
        for y in range(0, self.dsize[0], ystep):
            for x in range(0, self.dsize[1], xstep):
                testroi = testFeatureMap[y:y + ystep, x:x + xstep]
                sampleroi = sampleFeatureMap[y:y + ystep, x:x + xstep]
                testHist = self.calcHist(testroi)
                sampleHist = self.calcHist(sampleroi)
                k2 += np.sum((sampleHist - testHist) ** 2) / np.sum((sampleHist + testHist))
        return k2

    def predict(self, testImgName):
        '''
        预测函数
        :param dirName:样本图像文件夹路径
        :param testImgName: 测试图像文件名
        :return: 最相近图像名称
        '''
        testImg = testImgName

        k2List = []
        for img in self.imgList:
            k2 = self.compare(img, testImg)
            k2List.append(k2)
        order = np.argsort(k2List)
        return self.label[order[0]]