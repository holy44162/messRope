# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 08:35:05 2018

@author: Dufert
"""
import cv2 
import os,glob
import numpy as np
from PIL import Image
import scipy.signal as signal
import matplotlib.pyplot as plt
from skimage import exposure,transform
from skimage import feature as ft

'''
初始化read_feature_class类：
    choice：为特征选择 1 为hog 其他为 gabor
    pretreatment： 为预处理选择 1 为弱处理 其他为强处理
    只提供三个方法调用
    read_data_for_train：提供给训练集获取feature set，输入到train的路径即可
    read_data_for_cv_test：提供给交叉验证集和测试集获取feature set，输入到cv or test的路径即可
    read_data_for_predict：提供给实时预测获取image feature vector，输入摄像头获取的图像
example:
    read_obj = read_feature_class(3,3)
    train_path = 'G:/Computer Vision/Library/smallWinding/train_oc/'
    train_data,train_label,train_length = read_obj.read_data_for_train(train_path)
'''
class read_feature_class:
    
    def __init__(self,choice,pretreatment):
        '''
        初始化read_feature_class类：
            choice：为特征选择 1 为hog 其他为 gabor
            pretreatment： 为预处理选择 1 为弱处理 其他为强处理
        '''
        self.flag = True
        self.choice = choice
        self.pretreatment = pretreatment
        self.gabor_kernel = self.__gabor_kernel(17,6)
        self.kernel0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3))
        self.kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,3))
    
    
    def __gabor_kernel(self,ksize,num):
        '''
        获取gabor核
        '''
        filters = []
        for lamda in [11]:
             for length_width_ratio in [1]:         
                 for theta in [0]:#np.linspace(-np.pi*1/6, np.pi*1/6, num):
    #                 for ksize in range(ksize-2,ksize,2):
                     kern = cv2.getGaborKernel((ksize, ksize), #核size
                           2*length_width_ratio,     #高斯函数的标准差
                           theta,                    #法向平行条纹的方向
                           lamda,                    #波长的正弦因子
                           length_width_ratio,       #空间长宽比
                           0,                        #相位偏移
                           ktype=cv2.CV_32F)
                     kern /= kern.sum()
                     filters.append(kern)
        return filters
         
    
    def __uniformHandling(self,img):
        '''
        统一的预处理
        '''
#        gray_img = transform.resize(gray_img,(240,480))                                                                  #此为测试时使用的240*480尺寸，真实使用时注释掉
#        img_rotate = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = np.uint8(img)
        cut_img = img[288:547,438:753]
        gray_img = cv2.cvtColor(cut_img,cv2.COLOR_BGR2GRAY)/255
        gray_img = signal.medfilt(gray_img,3)
        
        return gray_img
        
    
    def __choosePretreatment(self,img):
        '''
        选择强弱预处理
        '''
        if self.pretreatment == 1:
            '''弱预处理'''
            gray_img = self.__uniformHandling(img)
            adap_img = exposure.equalize_adapthist(gray_img,(60,60))
            adap_img = exposure.equalize_adapthist(gray_img,(36,36))

        else:
            '''强预处理'''
            gray_img = self.__uniformHandling(img)
            adap_img = exposure.equalize_adapthist(gray_img,(120,120))
            adap_img = exposure.equalize_adapthist(adap_img,(32,32))
            adap_img = exposure.equalize_adapthist(adap_img,(4,8))
            adap_img = exposure.equalize_adapthist(adap_img,(32,32))
        
        return adap_img
    
    
    def __show(self,img):
        '''
        显示图像
        '''
        plt.figure(figsize=[7,7]),plt.imshow(img,cmap='gray'),plt.show()

    
    def __getHogFeature(self,img,predict_flag):
        '''
        Hog feature
        '''
        std_img = self.__choosePretreatment(img)
        
        if self.flag == True and predict_flag == False:
            self.flag = False
            self.__show(std_img)
            
#        down_img = cv2.pyrDown(std_img)
        features = ft.hog(std_img,
              orientations=9,  # number of binsa
              pixels_per_cell=(16,16), # pixel per cell
              cells_per_block=(3,3), # cells per blcok
              block_norm = 'L1', #  block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
              transform_sqrt = True, # power law compression (also known as gamma correction)
              feature_vector=True, # flatten the final vectors
              visualise=False) 
        
        hog_features = np.reshape(features,(len(features),1))

        return hog_features,std_img
    
    
    def __getGBHFeature(self,img,predict_flag):
        '''
        Gabor Binary Hog feature
        '''
        std_img = self.__choosePretreatment(img)
        
        if self.flag == True and predict_flag == False:
            self.flag = False
            self.__show(std_img)
        
        imgss = cv2.filter2D(std_img, cv2.CV_8UC3,self.gabor_kernel[0])
        dilate_img = cv2.dilate(imgss,self.kernel1)
        dilate_img = cv2.morphologyEx(dilate_img,cv2.MORPH_CLOSE,self.kernel1)
        down_img = cv2.pyrDown(dilate_img)

        hog_feature = ft.hog(down_img,
              orientations=9,  # number of binsa
              pixels_per_cell=(8,8), # pixel per cell
              cells_per_block=(2,2), # cells per blcok
              block_norm = 'L1', #  block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
              transform_sqrt = True, # power law compression (also known as gamma correction)
              feature_vector=True, # flatten the final vectors
              visualise=False) 
    
        gbh_features = np.reshape(hog_feature,(len(hog_feature),1))
        
        return gbh_features,dilate_img
    
    
    def __chooseFeatrue(self,predict_flag,img):
        '''
        选择特征返回
        '''
        if predict_flag == True:
            if self.choice == 1:
                features,down_img = self.__getHogFeature(img,predict_flag)
            else:
                features,down_img = self.__getGBHFeature(img,predict_flag)
            return features,down_img
        else:
            if self.choice == 1:
                features,_ = self.__getHogFeature(img,predict_flag)
            else:
                features,_ = self.__getGBHFeature(img,predict_flag)
            return features
    
    
    def read_data_for_train(self,path):
        '''
        读取训练集
        Input:
            path:训练集 路径
        Output：
            images：训练集 特征集
            label：训练集 标签
            length：训练集 每类的个数
        '''
        self.flag = True
        print('reading train data')
        suffix = '/*.jpg'
        label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
        images = []
        labels = []
        length = []
        predict_flag = False
        i = 1
        for index,folder in enumerate(label_dir):
            for image_path in glob.glob(folder+suffix):
                img = Image.open(image_path)
                features = self.__chooseFeatrue(predict_flag,img)
                i += 1
                if i % 100 == 0:
                    print(i)
                images.append(features)
                labels.append(index)
            length.append(len(labels))
        
        images = np.asarray(images,dtype=np.float32)
        label = np.asarray(labels,dtype=np.int32)
        label = -(label * 2 -1)
        length = np.asarray(length,dtype=np.int32)
        print('train data set read out')
        
        return images,label,length
    
    
    def read_data_for_cv_test(self,path):
        '''
        读取交叉验证集或测试集
        Input:
            path:交叉验证集或测试集 路径
        Output：
            images：交叉验证集或测试集 特征集
            label：交叉验证集或测试集 标签
            length：交叉验证集或测试集 每类的个数
        '''
        self.flag = True
        print('reading cv or test data')
        '''get data set hog feature'''
        suffix = '/*.jpg'
        label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
        images = []
        length = []
        predict_flag = False
        i = 1
        for index,folder in enumerate(label_dir):
            for image_path in glob.glob(folder+suffix):
                img = Image.open(image_path)
                features = self.__chooseFeatrue(predict_flag,img)
                images.append(features)
                i += 1
                if i % 100 == 0:
                    print(i)
        images = np.asarray(images,dtype=np.float32)
        
        '''get data set label and length'''
        try:
            f = open(path+'y_Test.txt')
            file_label = f.read()
            label = np.array(file_label.split(),np.int32)
            f.close()
        except:
            f = open(path+'y_CV.txt')
            file_label = f.read()
            label = np.array(file_label.split(),np.int32)
            f.close()
        
        length.append(len(label)-np.sum(label))
        length.append(np.sum(label))
        length = np.asarray(length,dtype=np.int32)
        label = -(label * 2 -1)
        
        print('cv or test data set read out')
        
        return images,label,length
    
    
    def read_data_for_predict(self,image):
        '''
        读取预测数据
        Input:
            image:摄像头图片数据
        Output：
            images：输入图片的特征向量
            down_img:提供给显示使用的图片
        '''
        images = []
        predict_flag = True
        
        """可能的风险：im的类型无法确定 要求输入类型image"""
        try:
            img = Image.fromarray(image.astype('uint8')).convert('RGB')
        except:
            print('输入类型不是uint8')
            img = image
        features,down_img = self.__chooseFeatrue(predict_flag,img)
        
        images.append(features)
        images = np.asarray(images,dtype=np.float32)
        images = np.reshape(images,[1,images.shape[1]])
        
        return images,down_img
    

# =============================================================================
# deprecated：
# gray_img = np.uint8(rotate_[:,:,0]*0.299 + rotate_[:,:,1]*0.587 + rotate_[:,:,2]*0.114)
#    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    cut_img = image[350:610,530:1050]
#     cut_img = image[385:664,550:1109]
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#    cut_img = clahe.apply(cut_img)
#    std_img = transform.resize(cut_img,(w,h))
#    features = ft.hog(std_img,
#          orientations=8,  # number of binsa
#          pixels_per_cell=(8,8), # pixel per cell
#          cells_per_block=(2,2), # cells per blcok
#          block_norm = 'L1', #  block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
#          transform_sqrt = True, # power law compression (also known as gamma correction)
#          feature_vector=True, # flatten the final vectors
#          visualise=False) 
# 
#    hog_feature = np.reshape(hog_feature,[int(len(hog_feature)/94),94])
#    hog_feature = np.transpose(hog_feature)
# 
#    pca1 = PCA(n_components=32)
#    pca1.fit(hog_feature)
#    X_new = pca1.transform(hog_feature)
#    gabor_pca_feature = np.reshape(X_new,(len(hog_feature)*32,1))
#    img_group.append(gabor_pca_feature)
#     
# def image_hog_for_predict(self,im):
#     #弱预处理
#     images = []
#     """可能的风险：im的类型无法确定 要求输入类型image"""
#     try:
#         img = Image.fromarray(im.astype('uint8')).convert('RGB')
#     except:
#         print('输入类型错误')
#     features = self.__chooseFeature(img)
#     images.append(features)
#     images = np.asarray(images,dtype=np.float32)
#     images = np.reshape(images,[1,images.shape[1]])
#     return images
# 
# 
# 
# def image_hog_for_predict(self,im):
#     
#     images = []
#     """可能的风险：im的类型无法确定 要求输入类型image"""
#     img = Image.fromarray(im.astype('uint8')).convert('RGB')
#     adap_img = self.__choosePretreatment(img)
#     std_img = signal.medfilt(adap_img, 3)
#     std_img = cv2.pyrDown(std_img)
# 
#     features = ft.hog(std_img,
#                       orientations=9,  # number of binsa
#                       pixels_per_cell=(8, 8),  # pixel per cell
#                       cells_per_block=(2, 2),  # cells per blcok
#                       block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
#                       transform_sqrt=True,  # power law compression (also known as gamma correction)
#                       feature_vector=True,  # flatten the final vectors
#                       visualise=False)
# 
#     new_feature = np.reshape(features, (len(features), 1))
#     images.append(new_feature)
#     images = np.asarray(images,dtype=np.float32)
#     images = np.reshape(images,[1,images.shape[1]])
#     return images,std_img
# 
# 
# def image_gbh_for_predict(self, im):
# 
#     images = []
#     """可能的风险：im的类型无法确定 要求输入类型image"""
#     img = Image.fromarray(im.astype('uint8')).convert('RGB')
#     adap_img = self.__choosePretreatment(img)
#     std_img = signal.medfilt(adap_img, 3)
# 
#     imgss = cv2.filter2D(std_img, cv2.CV_8UC3, self.gabor_kernel[0])
#     dilate_img = cv2.dilate(imgss, self.kernel0)
#     dilate_img = cv2.morphologyEx(dilate_img, cv2.MORPH_CLOSE, self.kernel1)
#     down_img = cv2.pyrDown(dilate_img)
# 
#     hog_feature = ft.hog(down_img,
#                          orientations=9,  # number of binsa
#                          pixels_per_cell=(8, 8),  # pixel per cell
#                          cells_per_block=(2, 2),  # cells per blcok
#                          block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
#                          transform_sqrt=True,  # power law compression (also known as gamma correction)
#                          feature_vector=True,  # flatten the final vectors
#                          visualise=False)
# 
#     gbh_feature = np.reshape(hog_feature, (len(hog_feature), 1))
# 
#     images.append(gbh_feature)
#     images = np.asarray(images, dtype=np.float32)
#     images = np.reshape(images, [1, images.shape[1]])
#     return images, dilate_img
# =============================================================================
