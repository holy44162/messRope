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

import matlab.engine

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

class KivyCamera(Image):

    def __init__(self, capture = None, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)

        self.eng = App.get_running_app().future.result()
        # self.eng.addpath('m:/files/files/phd/functions/messRopeFunctions', nargout=0)
        self.eng.addpath('e:/百度云同步盘/files/phd/functions/messRopeFunctions', nargout=0)

        # self.rectFilePathName = 'm:/files/files/phd/functions/messRopeFunctions/rect_anno.txt'
        # self.rotateFilePathName = 'm:/files/files/phd/functions/messRopeFunctions/angle_rotate.txt'

        self.rectFilePathName = './rect_anno.txt'
        self.rotateFilePathName = './angle_rotate.txt'

        video_files_path = './test2.mp4'
        self.capture = cv2.VideoCapture(video_files_path)

        return_value, frame = self.capture.read()
        if return_value:
            self.w, self.h = frame.shape[1], frame.shape[0]
            # bwRef = matlab.double([[1,2,3,4,5], [6,7,8,9,10]])
            self.bwRef = np.zeros((self.h, self.w))
            print(self.bwRef.shape)
            self.eng.load('bestPara.mat', nargout=0)
            eps = self.eng.workspace['epsilonOutput']
            print(eps)
            GMModelOutput = self.eng.workspace['GMModelOutput']
            GMModelOutputType = type(GMModelOutput)
            print(GMModelOutputType)
            self.eng.workspace['epsilonOutput'] = 9
            a = self.eng.eval('epsilonOutput+1')
            print(a)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('image', gray)
            data_list = gray.tolist()
            self.eng.fun_imshowPython(data_list, frame.shape[1], frame.shape[0], nargout=0)

            print(frame.shape)


            # vidFrame = matlab.double(list(frame))
            # self.eng.imshow(mat, nargout=0)

            # data1 = np.random.uniform(low = 0.0, high = 30000.0, size = (10,))
            # data1m = matlab.double(list(data1))
            # print(data1m)

            # frameType = type(frame)
            # print(frameType)

            # self.eng.imshow(frame, nargout=0)
            # GMModelOutput = matlab.object
            # bestPara, dataMLOutput, GMModelOutput, epsilonOutput = self.eng.fun_loadMatFile('bestPara.mat', nargout=4)
            # print(epsilonOutput)

        self.clockEvent = Clock.schedule_interval(self.update, 1.0 / 15)
        self.readFrequency = 30
        self.readCount = 0
        self.polygonLineThickness = 3
        self.messTag1 = 0
        self.messTag2 = 0

    # def start(self, capture, fps=30):
    #     self.capture = capture
    #     Clock.schedule_interval(self.update, 1.0 / fps)
    #
    # def stop(self):
    #     Clock.unschedule_interval(self.update)
    #     self.capture = None

    def update(self, dt):
        self.readCount += 1
        return_value, frame = self.capture.read()
        if return_value:
            # self.eng.fun_autoRecognizeByVideo(frame,self.rectFilePathName,\
            # self.rotateFilePathName,bestParaMats,self.bwRef)

            if (self.readCount % self.readFrequency) == 0:
                with open('./data_1.txt') as fp1:
                    tagFirst = fp1.readline().rstrip('\n')
                    line1 = '[color=ffff00]' + tagFirst + '[/color]'
                    App.get_running_app().root.ids.holyLabel1.text = line1

                    self.messTag1 = int(tagFirst)
                    if self.messTag1 == 1:
                        strPosFirst = fp1.readline().rstrip('\n')
                        strPosFirst = strPosFirst.split('\t')
                        self.pts1 = np.array([[int(strPosFirst[0]),int(strPosFirst[1])],\
                        [int(strPosFirst[2]),int(strPosFirst[3])],\
                        [int(strPosFirst[4]),int(strPosFirst[5])],\
                        [int(strPosFirst[6]),int(strPosFirst[7])]], np.int32)
                with open('./data_2.txt') as fp2:
                    tagSecond = fp2.readline().rstrip('\n')
                    line2 = '[color=ffff00]' + tagSecond + '[/color]'
                    App.get_running_app().root.ids.holyLabel2.text = line2

                    self.messTag2 = int(tagSecond)
                    if self.messTag2 == 1:
                        strPosSecond = fp2.readline().rstrip('\n')
                        strPosSecond = strPosSecond.split('\t')
                        self.pts2 = np.array([[int(strPosSecond[0]),int(strPosSecond[1])],\
                        [int(strPosSecond[2]),int(strPosSecond[3])],\
                        [int(strPosSecond[4]),int(strPosSecond[5])],\
                        [int(strPosSecond[6]),int(strPosSecond[7])]], np.int32)

            if self.messTag1 == 1:
                cv2.polylines(frame,[self.pts1],True,(0,0,255),self.polygonLineThickness)
            elif self.messTag2 == 1:
                cv2.polylines(frame,[self.pts2],True,(0,0,255),self.polygonLineThickness)

            if self.messTag1 == 1 or self.messTag2 == 1:
                App.get_running_app().root.ids.holyLabelMess.text = \
                '[b][color=ff0000]乱绳[/color][/b]'
                App.get_running_app().root.ids.holyEffect.effects = \
                [App.get_running_app().root.ids.holyEffect.effect_reference]
                # App.get_running_app().root.ids.holyLabelMess.font_size = \
                # App.get_running_app().root.font_scaling*60
            else:
                App.get_running_app().root.ids.holyLabelMess.text = \
                '[b][color=00ff00]正常[/color][/b]'

            texture = self.texture
            # w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != self.w or texture.height != self.h:
                self.texture = texture = Texture.create(size=(self.w, self.h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()
        else:
            self.capture.set(0, 0)
            # Clock.unschedule(self.clockEvent)
            print(self.capture)
            # self.eng.simple(nargout=0)
            # tf = self.eng.isprime(37)
            # print(tf)
            # self.capture = None

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

    future = matlab.engine.start_matlab(async=True)

    def build(self):
        Window.minimum_width = WINDOW_MIN_WIDTH
        Window.minimum_height = WINDOW_MIN_HEIGHT
        with open('./messropewin.kv', encoding='utf8') as f:
            self.messropeWin = Builder.load_string(f.read())
        return self.messropeWin

    def on_stop(self):
        if self.messropeWin.ids.qrcam.capture:
            print(self.messropeWin.ids.qrcam.capture)
            Clock.unschedule(self.messropeWin.ids.qrcam.clockEvent)
            self.messropeWin.ids.qrcam.capture.release()
            self.messropeWin.ids.qrcam.capture = None

if __name__ == '__main__':
    MessRopeApp().run()
