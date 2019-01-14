# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:38:14 2018

@author: Dufert
"""
import cv2
import warnings
import numpy as np
from skimage import morphology

warnings.filterwarnings('ignore')
'''
初始化multhread_predict_class类：
    choice：为特征选择 1 为hog 其他为 gabor
    只提供一个方法调用:  chooseFeatureOutput：返回识别结果，以及特征图片，在gabor时，提供坐标返回
example:
    frame：为获取的图像
    read_data = read_feature_class(1,1)
    clf = joblib.load("ocsvm_train_model_v53.m")
    pca1 = joblib.load("pca1_model_v0.m")
    predict_data = multhread_predict_class(1)
    predict_data.chooseFeatureOutput(frame,read_data,clf,pca1)
'''

class multhread_predict_class:


    def __init__(self,choice):
        '''
        初始化multhread_predict_class类：
            choice：为特征选择1 为hog,其他为gabor
        '''
        self.choice = choice
        img = cv2.imread('./img3.jpg')                                                                                   #当前为240*480尺寸在测试，改为img2为自由尺度模板
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,self.img = cv2.threshold(img_gray, 230, 1, cv2.THRESH_BINARY)


    def __std_output(self,value):
        '''
        标准化输出结果
        '''
        if value == 1:
            value = 0
        elif value == -1:
            value = 1
        else:
            print('error')
        return value


    def __predict_hog_image(self,image,read_data,ocsvm_model,pca_model):
        '''
        Input:
            image: 从ipcam获取的图像，尺寸要求: 720*1280*3
            read_ip_data: read_hog_class 类对象
            ocsvm_model: 训练得到的ocsvm模型
            pca_model: 训练得到的PCA模型

        Output：
            value: 识别结果 0为正常，1为乱绳
        '''

        image_node,image_thresh = read_data.read_data_for_predict(image)

        image_data = pca_model.transform(image_node)
        result = ocsvm_model.predict(image_data)
        value = self.__std_output(result[0])

        try:
            file_hog = open('data_2.txt','w')
            file_hog.write(str(value))
            file_hog.close()
        except:
            file_hog.close()
            print('Write txt error')

        return value,image_thresh


    def __predict_gabor_image(self,image,read_data,ocsvm_model,pca_model):
        '''
        Input:
            image_template: 提取卷扬部分的模板图片
            image: 从ipcam获取的图像，尺寸要求: 720*1280*3
            read_ip_data: read_hog_class 类对象
            ocsvm_model: 训练得到的ocsvm模型
            pca_model: 训练得到的PCA模型

        Output：
            value: 识别结果 0为正常，1为乱绳
            point: 位置
        '''
        img_row,img_col,_ = np.shape(image)
        image_node,image_thresh = read_data.read_data_for_predict(image)

        '''预测结果'''
        image_data = pca_model.transform(image_node)
        result = ocsvm_model.predict(image_data)
        value = self.__std_output(result[0])


        '''获取乱绳位置坐标
        存在的潜在问题：当获取的坐标不足4个时，将设定为无法写入，抛出 error  （未完成只是print）'''
        point = []
        # if value == 1:
        #     try:
        #         img_or = image_thresh | self.img
        #         img_or = (1 - img_or) * 255
        #         chull = np.uint8(morphology.convex_hull_image(img_or))
        #
        #         image, contours, hierarchy = cv2.findContours(chull, 2, 1)
        #         cnt = contours[0]
        #         hull = cv2.convexHull(cnt)
        #         point = hull[0:12:3, :, :]
        #
        #         '''计算的中心坐标'''
        #         angle = -4/180
        #         cen_row,cen_col = img_row/2,img_col/2
        #         cut_head_row,cut_head_col = 350,530
        #
        #         for i in range(4):
        #             [y,x] = point[i,0,:]#col,row
        #             u = int((y + cut_head_col - cen_row)*np.cos(angle) - (x + cut_head_row - cen_col)*np.sin(angle) + cen_row) #col
        #             v = int((y + cut_head_col - cen_row)*np.sin(angle) + (x + cut_head_row - cen_col)*np.cos(angle) + cen_col) #row
        #             point[i,0,:] = [v,u]
        #         point = np.reshape(point, (8))
        #     except:
        #         print('position error')
        # point = np.array(point,np.int32)


        # '''写txt  只给识别结果提供写入保证'''
        # try:
        #     file_gbh = open('data_2.txt', 'w')
        #     file_gbh.write(str(value)+'\n')
        #     file_gbh.close()
        # except:
        #     print('txt result error')
        #     file_gbh.close()
        # try:
        #     file_gbh = open('data_2.txt', 'a')
        #     for i in range(8):
        #         file_gbh.write(str(point[i])+' ')
        #     file_gbh.close()
        # except:
        #     print('txt position error')
        #     file_gbh.close()

        return value,point,image_thresh


    def chooseFeatureOutput(self,image,read_data,ocsvm_model,pca_model):
        '''
        提供给外部调用的接口
        Input：
            image：摄像头获取的图像
            read_data: read_feature_class对象
            ocsvm_model:训练得到的ocsvm模型
            pca_model:训练得到的PCA模型
        Output：
        case 1：
            value：识别结果 1为乱绳，0为正常
            image_thresh：获取提取hog之前的图像以供显示
        case 2：
            value：识别结果 1为乱绳，0为正常
            point：乱绳区域坐标
            image_thresh：获取gabor二值化图像以供显示
        '''
        if self.choice == 1:
            value,image_thresh = self.__predict_hog_image(image,read_data,ocsvm_model,pca_model)
            return value,image_thresh
        else:
            value,point,image_thresh = self.__predict_gabor_image(image,read_data,ocsvm_model,pca_model)
            return value,point,image_thresh
