# !/usr/bin/env python 
# -*- coding:utf-8 -*-
#返回 起始坐标，终点坐标，实时坐标，还有map
import cv2
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import copy

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Color.
params.filterByColor = True
params.blobColor = 0

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(320, 240))
stream = camera.capture_continuous(rawCapture, format="bgr",
                                   use_video_port=True, burst=True)
camera.close()

vs = PiVideoStream().start()


class picMag(object):
    msg = dict()

    def __init__(self, map_w,map_h):
        self.map_w = map_w
        self.map_h = map_h
        self.msg['action'] = 0
    def getPic(self):
        return  vs.read()

    def binary(self):
        self.msg['originalPic'] = self.getPic()
        if self.msg['action']  == 0:
            self.msg['showPath'] = copy.deepcopy(self.msg['originalPic'])
            self.msg['showPos'] = copy.deepcopy(self.msg['originalPic'])
            image_binary = cv2.cvtColor(self.msg['originalPic'], cv2.COLOR_BGR2GRAY)
            image_binary = cv2.medianBlur(image_binary, 5)
            ret, image_binary = cv2.threshold(image_binary, 127, 255, cv2.THRESH_BINARY)
            image_binary = cv2.medianBlur(image_binary, 3)
        if self.msg['action'] == 3:

            cv2.rectangle(self.msg['originalPic'], (int(self.msg['tablet_x'] + self.msg['ball_end_x'] - 1),
                                                    int(self.msg['tablet_y'] + self.msg['ball_end_y'] - 1)),
                          (int(self.msg['tablet_x'] + self.msg['ball_end_x'] + self.msg['ball_end_w'] + 1),
                           int(self.msg['tablet_y'] + self.msg['ball_end_y'] + self.msg['ball_end_h'] + 1)),
                          (0, 0, 0), -1)
            cv2.circle(self.msg['originalPic'], (
            int(self.msg['tablet_x'] + self.msg['ball_start_x']), int(self.msg['tablet_y'] + self.msg['ball_start_y'])),
                       int(self.msg['ball_r'] + 11),
                       (0, 0, 0), -1)

            self.msg['image_end'] = copy.deepcopy(self.msg['originalPic'])
            image_binary = cv2.cvtColor(self.msg['originalPic'], cv2.COLOR_BGR2GRAY)
            ret, image_binary = cv2.threshold(image_binary, 127, 255, cv2.THRESH_BINARY)
        elif self.msg['action'] == 2 or self.msg['action'] == 4:
            cv2.rectangle(self.msg['originalPic'], (int(self.msg['tablet_x'] + self.msg['ball_end_x'] - 1),
                                                    int(self.msg['tablet_y'] + self.msg['ball_end_y'] - 1)),
                          (int(self.msg['tablet_x'] + self.msg['ball_end_x'] + self.msg['ball_end_w'] + 1),
                           int(self.msg['tablet_y'] + self.msg['ball_end_y'] + self.msg['ball_end_h'] + 1)),
                          (0, 0, 0), -1)
            image_binary = cv2.cvtColor(self.msg['originalPic'], cv2.COLOR_BGR2GRAY)
            ret, image_binary = cv2.threshold(image_binary, 127, 255, cv2.THRESH_BINARY)
        self.msg['binaryPic'] = cv2.bitwise_not(image_binary);

        pass
    def findTablet(self):
        average_mag,num_x , num_y, num_h,num_w = 0 , 0 , 0 ,0 , 0
        while True:
            self.binary()
            image_find_path = copy.deepcopy(self.msg['binaryPic'])
            # 构造一个3×3的结构元素
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilate = cv2.dilate(image_find_path, element)
            erode = cv2.erode(image_find_path, element)

            # 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
            result = cv2.absdiff(dilate, erode);
            result = cv2.medianBlur(result, 5)
            # 上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
            retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY);
            # 反色，即对二值图每个像素取反
            result = cv2.bitwise_not(result);
            resultOK = copy.deepcopy(result)
            imag, contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            self.msg['tablet_x'], self.msg['tablet_y'], self.msg['tablet_w'], self.msg['tablet_h'] = cv2.boundingRect(
                contours[0])
            for i in range(0, len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                if w > 190 and h > 190 and w < 220 and h < 220 and x > 50 and x < 70 and y > 10 and y < 30:
                    self.msg['tablet_x'] = x
                    self.msg['tablet_y'] = y
                    self.msg['tablet_w'] = w
                    self.msg['tablet_h'] = h
                    # 提取出平板大小

                    average_mag =average_mag + 1
                    num_x = num_x + self.msg['tablet_x']
                    num_y = num_y + self.msg['tablet_y']
                    num_w = num_w + self.msg['tablet_w']
                    num_h = num_h + self.msg['tablet_h']
            if average_mag >= 10:
                self.msg['tablet_x'] = int(num_x / average_mag)
                self.msg['tablet_y'] = int(num_y / average_mag)
                self.msg['tablet_w'] = int(num_w / average_mag)
                self.msg['tablet_h'] = int(num_h / average_mag)

                break

    def findStart(self):
        average_mag , num_x, num_y, num_r = 0, 0, 0 , 0
        while True:
            self.binary()
            image_find_ball = self.image_binary[self.msg['tablet_y']:(self.msg['tablet_y'] + self.msg['tablet_h']),
                              self.msg['tablet_x']:(self.msg['tablet_x'] + self.msg['tablet_w'])]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 矩形结构
            dilation = cv2.dilate(image_find_ball, kernel, iterations=2)
            # dilation = cv2.erode(dilation, kernel,iterations = 1)
            # dilation = cv2.medianBlur(dilation,5)
            circles = cv2.HoughCircles(dilation, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=5, minRadius=2, maxRadius=9)
            if circles is not None:
                for circle in circles[0, :]:
                    average_mag = average_mag+1
                    num_x = num_x + int(circle[0])
                    num_y = num_y + int(circle[1])
                    num_r = num_r + int(circle[2])
                    # 在找球图用指定颜色标记出圆的位置
                    # img = cv2.circle(image_find_ball,(temp_ball_x,temp_ball_y),temp_ball_r+2,(0,0,255),-1)
            if average_mag >= 10:
                self.msg['ball_start_x'] = int(num_x / average_mag)
                self.msg['ball_start_y'] = int(num_y / average_mag)
                self.msg['ball_r'] = int(num_r / average_mag)
                break

    def findEnd(self):

        average_mag, num_x, num_y, num_h, num_w = 0, 0, 0, 0, 0
        while True:
            self.binary()
            image_find_end = self.msg['originalPic'][self.msg['tablet_y']:(self.msg['tablet_y'] + self.msg['tablet_h']),
                             self.msg['tablet_x']:(self.msg['tablet_x'] + self.msg['tablet_w'])]

            lower_blue = np.array([0, 43, 46])
            upper_blue = np.array([20, 255, 255])
            hsv = cv2.cvtColor(image_find_end, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 7)
            # get mask
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            # cv2.imshow('O2', mask)
            # detect blue
            res = cv2.bitwise_and(image_find_end, image_find_end, mask=mask)
            res = cv2.medianBlur(res, 3)
            res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            retval_end, result_end = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY);
            result_end = cv2.medianBlur(result_end, 3)
            # cv2.imshow('OK222', result_end)
            image_end_result, contours, hierarchy = cv2.findContours(result_end, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # temp_end_x, temp_end_y, temp_end_w, temp_end_h = cv2.boundingRect(contours[0])
            if contours is not None:
                for i in range(0, len(contours)):
                    x, y, w, h = cv2.boundingRect(contours[i])
                    if w < 30 and h < 30 and w > 5 and h > 5:
                        average_mag = average_mag + 1
                        num_x = gnum_x + x
                        num_y = num_y + y
                        num_w = num_w + w
                        num_h = num_h + h
            if average_mag >= 10:
                self.msg['ball_end_x'] = int(num_x / g_average_mag)
                self.msg['ball_end_y'] = int(num_y / g_average_mag)
                self.msg['ball_end_w'] = int(num_w / g_average_mag)
                self.msg['ball_end_h'] = int(num_h / g_average_mag)
                break
    def findBall(self):
        self.binary()
        image_find_ball = self.image_binary[self.msg['tablet_y']:(self.msg['tablet_y'] + self.msg['tablet_h']),
                          self.msg['tablet_x']:(self.msg['tablet_x'] + self.msg['tablet_w'])]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilation = cv2.dilate(image_find_ball, kernel, iterations=2)
        circles = cv2.HoughCircles(dilation, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=3, minRadius=2, maxRadius=9)
        if circles is not None:
            for circle in circles[0, :]:

                ball_x = int(circle[0])
                ball_y = int(circle[1])
                return ball_x, ball_y, True
        else:
            return 0,0,False


    def getMap(self):
        self.binary()
        image_find_path = copy.deepcopy(self.image_binary)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilate = cv2.dilate(image_find_path, element)
        erode = cv2.erode(image_find_path, element)

        result = cv2.absdiff(dilate, erode);
        result = cv2.medianBlur(result, 5)

        retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY);
        result = cv2.bitwise_not(result);
        resultOK = copy.deepcopy(result)
        imag, contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.msg['tabletPic'] = resultOK[self.msg['tablet_y']:(self.msg['tablet_y'] + self.msg['tablet_h']), self.msg['tablet_x']:(self.msg['tablet_x'] + self.msg['tablet_w'])]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
        erosion = cv2.erode(self.msg['tabletPic'], kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=2)
        self.msg['tabletPic'] = cv2.medianBlur(dilation, 3)

    def magMap(self,map_data_mag):
        kernel = np.ones((7, 7), np.uint8)
        size = self.msg['tabletPic'].shape
        self.msg['ratio_x'] = size[0] / (self.map_w)
        self.msg['ratio_y'] = size[1] / (self.map_h)
        self.msg['image_res'] = cv2.resize(self.msg['tabletPic'], (map_w, map_h), interpolation=cv2.INTER_AREA)
        self.msg['path_mag'] = cv2.medianBlur(self.msg['tabletPic'], 3)

        for i in range(self.map_w):
            for j in range(self.map_h):
                if self.msg['image_res'][i][j] < 200:
                    self.msg['image_res'][i][j] = 0
                else:
                    self.msg['image_res'][i][j] = 255
        for i in range(self.map_w + 2):
            for j in range(self.map_h + 2):
                if i == 0:
                    map_data_mag[i][j] = '#'
                elif i == self.map_w + 1:
                    map_data_mag[i][j] = '#'
                else:
                    if j == 0 or j == self.map_h + 1:
                        map_data_mag[i][j] = '#'
                    else:
                        if self.msg['image_res'][i - 1][j - 1] == 0:
                            map_data_mag[i][j] = '#'
                        else:
                            map_data_mag[i][j] = '.'




