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

        video_files_path = './test2.mp4'
        self.capture = cv2.VideoCapture(video_files_path)
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
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()
        else:
            self.capture.set(0, 0)
            # Clock.unschedule(self.clockEvent)
            print(self.capture)
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
