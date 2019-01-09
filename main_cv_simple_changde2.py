from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.properties import NumericProperty
import numpy as np
import cv2
from kivy.uix.effectwidget import EffectWidget, EffectBase

# import matlab.engine

# from pathlib import Path

from threading import Thread

# from PIL import Image as ImagePillow
from collections import deque as dq
from sklearn.externals import joblib
from read_feature_class import read_feature_class
from multhread_predict_class import multhread_predict_class

dp = dq([0, 0, 0], maxlen=3)
read_data = read_feature_class(1,1)
predict_data = multhread_predict_class(1)
clf = joblib.load(r"d:\backup\project\changde_winding_code3\changde_hog_ocsvm_train_model_v1.m")
pca1 = joblib.load(r"d:\backup\project\changde_winding_code3\changde_hog_pca1_model_v1.m")

WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600

# The effect string is glsl code defining an effect function.
effect_string = '''
vec4 effect(vec4 color, sampler2D texture, vec2 tex_coords, vec2 coords)
{
    // Note that time is a uniform variable that is automatically
    // provided to all effects.
    float red = color.x * abs(sin(time*2.0));
    float green = color.y;  // No change
    float blue = color.z * (1.0 - abs(sin(time*2.0)));
    return vec4(red, green, blue, color.w);
}
'''

class DemoEffect(EffectWidget):
    def __init__(self, *args, **kwargs):
        self.effect_reference = EffectBase(glsl=effect_string)
        super(DemoEffect, self).__init__(*args, **kwargs)

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                self.stream.set(0, 0)
                print(self.stream)

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class KivyCamera(Image):

    def __init__(self, capture = None, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)

        # self.eng = App.get_running_app().future.result()
        # self.eng.addpath('m:/files/files/phd/functions/messRopeFunctions', nargout=0)
        # self.eng.addpath('e:/百度云同步盘/files/phd/functions/messRopeFunctions', nargout=0)

        # self.rectFilePathName = 'm:/files/files/phd/functions/messRopeFunctions/rect_anno.txt'
        # self.rotateFilePathName = 'm:/files/files/phd/functions/messRopeFunctions/angle_rotate.txt'

        # self.rectFilePathName = 'rect_anno.txt'
        # self.rotateFilePathName = 'angle_rotate.txt'

        # video_files_path = 'd:/data_seq/changdeWinding/winding2/test_changde2.mp4'
        video_files_path = 'rtsp://admin:cl123456@192.168.1.120/Streaming/Channels/201'
        # self.capture = cv2.VideoCapture(video_files_path)
        self.vs = WebcamVideoStream(video_files_path).start()

        # return_value, frame = self.capture.read()
        # if return_value:
        #     self.w, self.h = frame.shape[1], frame.shape[0]
        #
        #     # bwRef = matlab.double([[1,2,3,4,5], [6,7,8,9,10]])
        #     # self.bwRef = np.zeros((self.h, self.w))
        #     # print(self.bwRef.shape)
        #     # self.eng.load('bestPara.mat', nargout=0)
        #     #
        #     # bestPara = self.eng.workspace['bestPara']
        #     # dataMLOutput = self.eng.workspace['dataMLOutput']
        #     # GMModelOutput = self.eng.workspace['GMModelOutput']
        #     # epsilonOutput = self.eng.workspace['epsilonOutput']
        #
        #     print(frame.shape)
        #     # t1 = time.time()
        #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     # gray = cv2.resize(gray,(10,20))
        #     # # data_list_matlab = matlab.double(gray)
        #     # # print(type(data_list_matlab))
        #     # # print(data_list_matlab.size)
        #     # # print(data_list_matlab)
        #     # A = matlab.double([[1,2,3,4,5], [6,7,8,9,10]])
        #     # print(A)
        #     # # data_list = gray.tolist()
        #     # # flat_list = [item for sublist in data_list for item in sublist]
        #     # # self.eng.imshow(data_list_matlab, nargout=0)
        #     # # newimg = cv2.resize(gray,(10,20))
        #     # # print(newimg.shape)
        #     # # print(type(newimg))
        #     # # print(newimg)
        #     # # data_list = newimg.tolist()
        #     # # print(len(data_list))
        #     # # print(type(data_list))
        #     # # print(data_list)
        #     # # flat_list = [item for sublist in data_list for item in sublist]
        #     # # print(len(flat_list))
        #     # # print(type(flat_list))
        #     # # print(flat_list)
        #     # # self.eng.fun_imshowPython(data_list, frame.shape[1], frame.shape[0], nargout=0)
        #     # # data_list_matlab = matlab.uint8(data_list)
        #     # # messTagMatlab, messPosMatlab = self.eng.fun_autoRecognizeByVideoPython1(data_list_matlab,self.rectFilePathName,\
        #     # # self.rotateFilePathName,bestPara,dataMLOutput,GMModelOutput,epsilonOutput,\
        #     # # frame.shape[1],frame.shape[0],nargout=2)
        #     # # print(messTagMatlab)
        #     # # print(messPosMatlab)
        #     # elapsed1 = time.time() - t1
        #     # print(elapsed1)
        #
        #     # eps = self.eng.workspace['epsilonOutput']
        #     # print(eps)
        #     # GMModelOutput = self.eng.workspace['GMModelOutput']
        #     # GMModelOutputType = type(GMModelOutput)
        #     # print(GMModelOutputType)
        #     # self.eng.workspace['epsilonOutput'] = 9
        #     # a = self.eng.eval('epsilonOutput+1')
        #     # print(a)
        #
        #     # t1 = time.time()
        #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     # # cv2.imshow('image', gray)
        #     # data_list = gray.tolist()
        #     # self.eng.fun_imshowPython(data_list, frame.shape[1], frame.shape[0], nargout=0)
        #     # elapsed1 = time.time() - t1
        #     # print(elapsed1)
        #     # print(frame.shape)
        #
        #
        #
        #     # t2 = time.time()
        #     # bChannel,gChannel,rChannel = cv2.split(frame)
        #     # data_listR = rChannel.tolist()
        #     # data_listG = gChannel.tolist()
        #     # data_listB = bChannel.tolist()
        #     # self.eng.fun_imshowColor(data_listR, data_listG, data_listB, \
        #     # frame.shape[1], frame.shape[0], nargout=0)
        #     # elapsed2 = time.time() - t2
        #     # print(elapsed2)
        #
        #     # imPillow = ImagePillow.fromarray(frame)
        #     # image_mat = matlab.uint8(list(imPillow.getdata()))
        #     # image_mat.reshape((imPillow.size[0], imPillow.size[1], 3))
        #     # self.eng.fun_imshowPillow(image_mat, nargout=0)
        #
        #
        #     # vidFrame = matlab.double(list(frame))
        #     # self.eng.imshow(mat, nargout=0)
        #
        #     # data1 = np.random.uniform(low = 0.0, high = 30000.0, size = (10,))
        #     # data1m = matlab.double(list(data1))
        #     # print(data1m)
        #
        #     # frameType = type(frame)
        #     # print(frameType)
        #
        #     # self.eng.imshow(frame, nargout=0)
        #     # GMModelOutput = matlab.object
        #     # bestPara, dataMLOutput, GMModelOutput, epsilonOutput = self.eng.fun_loadMatFile('bestPara.mat', nargout=4)
        #     # print(epsilonOutput)

        # Thread(target=self.updateFrames, args=()).start()
        self.clockEvent = Clock.schedule_interval(self.update, 1.0 / 25)
        # self.clockEvent = Clock.schedule_once(self.update, 5)
        # self.readFrequency = 30
        # self.readCount = 0
        # self.polygonLineThickness = 3
        # self.messTag1 = 0
        # self.messTag2 = 0

    # def start(self, capture, fps=30):
    #     self.capture = capture
    #     Clock.schedule_interval(self.update, 1.0 / fps)
    #
    # def stop(self):
    #     Clock.unschedule_interval(self.update)
    #     self.capture = None

    # def updateFrames(self):
	# 	# keep looping infinitely until the thread is stopped
    #     while True:
    #         (self.grabbed, self.frame) = self.capture.read()
    #         if not self.grabbed:
    #             self.capture.set(0, 0)
    #             print(self.capture)
    #
    # def readFrame(self):
    #     return self.frame

    def update(self, dt):
        if self.vs.grabbed:
            frame = self.vs.read()
            # add predict code here
            tagMess,_ = predict_data.chooseFeatureOutput(frame,read_data,clf,pca1)
            line1 = '[color=ffff00]' + str(tagMess) + '[/color]'
            App.get_running_app().root.ids.holyLabel1.text = line1
            dp.appendleft(tagMess)
            if np.sum(dp) == 3:
                App.get_running_app().root.ids.holyLabelMess.text = \
                '[b][color=ff0000]乱绳[/color][/b]'
                App.get_running_app().root.ids.holyEffect.effects = \
                [App.get_running_app().root.ids.holyEffect.effect_reference]
            else:
                App.get_running_app().root.ids.holyLabelMess.text = \
                '[b][color=00ff00]正常[/color][/b]'

            # if tagMess == 1:
            #     App.get_running_app().root.ids.holyLabelMess.text = \
            #     '[b][color=ff0000]乱绳[/color][/b]'
            #     App.get_running_app().root.ids.holyEffect.effects = \
            #     [App.get_running_app().root.ids.holyEffect.effect_reference]
            #     # App.get_running_app().root.ids.holyLabelMess.font_size = \
            #     # App.get_running_app().root.font_scaling*60
            # else:
            #     App.get_running_app().root.ids.holyLabelMess.text = \
            #     '[b][color=00ff00]正常[/color][/b]'
            # end of addition

            # cv2.imshow("Frame", frame)
            w, h = frame.shape[1], frame.shape[0]
            texture = self.texture

            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()
        # while True:
        #     frame = self.readFrame()
        #     print(frame.shape)
        #     # return_value, frame = self.capture.read()
        #     # if return_value:
        #     texture = self.texture
        #
        #     if not texture or texture.width != self.w or texture.height != self.h:
        #         self.texture = texture = Texture.create(size=(self.w, self.h))
        #         texture.flip_vertical()
        #     texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
        #     self.canvas.ask_update()
        # # else:
        # #     self.capture.set(0, 0)
        # #     # Clock.unschedule(self.clockEvent)
        # #     print(self.capture)
        # #     # self.eng.simple(nargout=0)
        # #     # tf = self.eng.isprime(37)
        # #     # print(tf)
        # #     # self.capture = None

class MessRopeRoot(Screen):

    font_scaling = NumericProperty()

    def on_size(self, *args):
        self.font_scaling = min(Window.width/WINDOW_MIN_WIDTH, Window.height/WINDOW_MIN_HEIGHT)
        # self.ids.holyLabel.text = '[color=ffff00]changed[/color]'

    def showcase_boxlayout(self, layout):
        pass

    # def dostart(self, *largs):
    #     global capture
    #     video_files_path = './data/test1.mp4'
    #     capture = cv2.VideoCapture(video_files_path)
    #     self.ids.qrcam.start(capture)
    #
    # def doexit(self):
    #     global capture
    #     if capture != None:
    #         capture.release()
    #         capture = None
    #     EventLoop.close()


class MessRopeApp(App):

    # future = matlab.engine.start_matlab(async=True)

    def build(self):
        Window.minimum_width = WINDOW_MIN_WIDTH
        Window.minimum_height = WINDOW_MIN_HEIGHT
        with open('./messropewinchangde.kv', encoding='utf8') as f:
            self.messropeWin = Builder.load_string(f.read())
        return self.messropeWin

    def on_stop(self):
        if self.messropeWin.ids.qrcam.vs.grabbed:
            print(self.messropeWin.ids.qrcam.vs.stream)
            Clock.unschedule(self.messropeWin.ids.qrcam.clockEvent)
            self.messropeWin.ids.qrcam.vs.stop()
            self.messropeWin.ids.qrcam.vs.stream.release()
            self.messropeWin.ids.qrcam.vs.stream = None

if __name__ == '__main__':
    MessRopeApp().run()
