from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import imutils
import serial
import time
import cv2
import copy
import math
import serial
import operator

#因为python里string不能直接改变某一元素，所以用test_map来存储搜索时的地图
test_map = []


Line = []

map_path =   [[0 for x in range(500)] for y in range(500)]
map_path_piont =   [[0 for x in range(500)] for y in range(500)]
map_data_mag =   []


#port = serial.Serial("/dev/ttyAMA0", baudrate=115200, timeout=1.0)
# Send_data
Uart_buf = [0x55,0x00,0xAA,0x00,0xAA,0x00,0xAA,0x00,0xAA,0x00,0xAA,0xAA,0xAA,0xAA]
# Setup Usart
port = serial.Serial("/dev/ttyAMA0", baudrate=115200,timeout=0)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255


# Filter by Color.
params.filterByColor = True                                
params.blobColor = 0


data = [[0 for x in range(25)] for y in range(25)]

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else :
	detector = cv2.SimpleBlobDetector_create(params)

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(320, 240))
stream = camera.capture_continuous(rawCapture, format="bgr",
	use_video_port=True, burst=True)
camera.close()

vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()


global g_start_search_flag ,g_start_search_flag,g_average_mag,g_static_num_x,g_static_num_y,\
       g_static_num_w,g_static_num_h,g_static_num_r,g_start_map_flag
global ball_start_x,ball_start_y,ball_r
global ball_end_x,ball_end_y,ball_end_w,ball_end_h
global ball_x,ball_y
global ball_last_x,ball_last_y
global ball_move_i,flag_move
global ball_move_x_ok,ball_move_y_ok
ball_move_i = 0
ball_last_x = 0
ball_last_y = 0
flag_move = 0
ball_move_x_ok = 0
ball_move_y_ok = 0
    #真实图片与处理图片缩放比

global ratio_x,ratio_y
global path_len    ,img_path
global tablet_x, tablet_y, tablet_w, tablet_h
global g_ball_start_flag,g_ball_end_flag,g_tablet_flag,run_time
global g_start_time,g_end_time

g_start_map_flag = 0
g_start_time= 0
g_end_time= 0
g_ball_start_flag = 0
g_ball_end_flag = 0
g_tablet_flag = 0
run_time = 0
path_len = 0
ball_start_x=0
ball_start_y=0
ball_r = 0
ball_end_x=0
ball_end_y=0
ball_end_w=0
ball_end_h=0
ball_x=0
ball_y=0
ratio_x=0
ratio_y=0
tablet_x=0
tablet_y=0
tablet_w=0
tablet_h=0
g_start_search_flag = 0
g_average_mag = 0
g_static_num_x= 0
g_static_num_y= 0
g_static_num_w= 0
g_static_num_h= 0
g_static_num_r= 0
global map_w,map_h,num_i
num_i = 0
map_w =  30
map_h =  30
tm =   [[0 for x in range(map_w+2)] for y in range(map_h+2)]
def find_curve():
    
#小球起始坐标，结束坐标
    global ball_start_x,ball_start_y,ball_r,ball_move_y_ok,ball_move_x_ok,num_i
    global ball_end_x,ball_end_y,ball_end_w,ball_end_h
    global ball_x,ball_y,ball_move_i,flag_move
    #真实图片与处理图片缩放比
    global ratio_x,ratio_y
    global g_start_time,g_end_time 
    global tablet_x, tablet_y, tablet_w, tablet_h
    global ball_last_x,ball_last_y
    global g_average_mag,g_static_num_x,g_static_num_y,\
           g_static_num_w,g_static_num_h,g_static_num_r
    
    global image_end,img_path
    
    global g_start_search_flag,g_start_map_flag
    
    global image_tablet    
    global image_original,image_binary,image_find_end,start_time
    global g_ball_start_flag,g_ball_end_flag,g_tablet_flag
       
    #起始时间记录
    start_time = time.time()
    image_original = vs.read()
    
    #cv2.imshow('image',image)
    if g_start_search_flag == 3:
        
        #image_end = copy.deepcopy(image_original)
        cv2.rectangle(image_original,(int(tablet_x+ball_end_x-1),int(tablet_y+ball_end_y-1)),\
                (int(tablet_x+ball_end_x+ball_end_w+1),int(tablet_y+ball_end_y+ball_end_h+1)), (0,0,0), -1)
        cv2.circle(image_original,(int(tablet_x+ball_start_x),int(tablet_y+ball_start_y)),int(ball_r+11),(0,0,0),-1)
            
        image_find_end = copy.deepcopy(image_original)
        image_end = copy.deepcopy(image_original)
        #二值化处理    
        image_binary = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)
        #image_binary = cv2.medianBlur(image_binary,5)
        ret,image_binary = cv2.threshold(image_binary,127,255,cv2.THRESH_BINARY)
        #自适应二值化，没有固定阈值好    
        #image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #                           cv2.THRESH_BINARY,9S,5)    
        #image_binary = cv2.medianBlur(image_binary,3)
        #反色处理
        image_binary = cv2.bitwise_not(image_binary);
        
        
    elif g_start_search_flag == 2  : 
        cv2.rectangle(image_original,(int(tablet_x+ball_end_x-1),int(tablet_y+ball_end_y-1)),\
                (int(tablet_x+ball_end_x+ball_end_w+1),int(tablet_y+ball_end_y+ball_end_h+1)), (0,0,0), -1)
            
        #二值化处理    
        image_binary = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)
        #image_binary = cv2.medianBlur(image_binary,5)
        ret,image_binary = cv2.threshold(image_binary,127,255,cv2.THRESH_BINARY)
        #自适应二值化，没有固定阈值好    
        #image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #                           cv2.THRESH_BINARY,9S,5)    
        #image_binary = cv2.medianBlur(image_binary,3)
        #反色处理
        image_binary = cv2.bitwise_not(image_binary);
    elif g_start_search_flag == 4: 
        cv2.rectangle(image_original,(int(tablet_x+ball_end_x-1),int(tablet_y+ball_end_y-1)),\
                (int(tablet_x+ball_end_x+ball_end_w+1),int(tablet_y+ball_end_y+ball_end_h+1)), (0,0,0), -1)
            
        #二值化处理    
        image_binary = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)
        #image_binary = cv2.medianBlur(image_binary,5)
        ret,image_binary = cv2.threshold(image_binary,127,255,cv2.THRESH_BINARY)
      
        image_binary = cv2.bitwise_not(image_binary);
        

    else :
        img_path = copy.deepcopy(image_original)
        image_find_end = copy.deepcopy(image_original)
        image_end = copy.deepcopy(image_original)
        #二值化处理    
        image_binary = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)
        image_binary = cv2.medianBlur(image_binary,5)
        ret,image_binary = cv2.threshold(image_binary,127,255,cv2.THRESH_BINARY)
        #自适应二值化，没有固定阈值好    
        #image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #                           cv2.THRESH_BINARY,9S,5)    
        image_binary = cv2.medianBlur(image_binary,3)
        #反色处理
        image_binary = cv2.bitwise_not(image_binary);
        

    
    '''
    路径处理，主要找出真实平板图片，去掉边沿多余图像,捕获平板坐标

    '''
    if g_start_search_flag == 0:    
        image_find_path = copy.deepcopy(image_binary)
        #构造一个3×3的结构元素   
        element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))  
        dilate = cv2.dilate(image_find_path, element)  
        erode = cv2.erode(image_find_path, element)  

        #将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像  
        result = cv2.absdiff(dilate,erode);    
        result = cv2.medianBlur(result,5)
        #上面得到的结果是灰度图，将其二值化以便更清楚的观察结果  
        retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY);
        #反色，即对二值图每个像素取反  
        result = cv2.bitwise_not(result);
        resultOK = copy.deepcopy(result)
        imag, contours, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        tablet_x, tablet_y, tablet_w, tablet_h = cv2.boundingRect(contours[0])  
        for i in range(0,len(contours)):  
            x, y, w, h = cv2.boundingRect(contours[i])
            if w>190 and h>190 and w<220 and h<220 and x>50 and x<70 and y>10 and y <30:
                tablet_x = x
                tablet_y = y
                tablet_w = w
                tablet_h = h
                #cv2.rectangle(result, (temp_x,temp_y), (temp_x+temp_w,temp_y+temp_h), (255,255,255), 5)
        #cv2.rectangle(imag, (50,50), (x3+w,y3+h), (255,255,255), 5)
        #提取出平板大小
        image_tablet = resultOK[ tablet_y:(tablet_y+tablet_h),tablet_x:(tablet_x+tablet_w)]
  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
        erosion = cv2.erode(image_tablet,kernel,iterations = 2) 
        dilation = cv2.dilate(erosion, kernel,iterations = 2)
        image_tablet = cv2.medianBlur(dilation,3)
        g_average_mag=g_average_mag+1     
        
        g_static_num_x= g_static_num_x + tablet_x
        g_static_num_y= g_static_num_y + tablet_y
        g_static_num_w= g_static_num_w + tablet_w
        g_static_num_h= g_static_num_h + tablet_h
        if g_average_mag == 10:
            g_start_search_flag = 1
            tablet_x = int(g_static_num_x/10)
            tablet_y = int(g_static_num_y/10)
            tablet_w = int(g_static_num_w/10)
            tablet_h = int(g_static_num_h/10)           
            g_average_mag =   0   
            g_static_num_x= 0
            g_static_num_y= 0
            g_static_num_w= 0
            g_static_num_h= 0
            print("Map---------OK")
                
     
        cv2.imshow('image_tablet', image_tablet)
    '''
    滚球坐标寻找，包括其实坐标

    '''
    if g_start_search_flag == 2:
        
        image_find_ball = image_binary[ tablet_y:(tablet_y+tablet_h),tablet_x:(tablet_x+tablet_w)]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 矩形结构
        
        dilation = cv2.dilate(image_find_ball, kernel,iterations = 2)
        #dilation = cv2.erode(dilation, kernel,iterations = 1)
        #dilation = cv2.medianBlur(dilation,5)

        circles= cv2.HoughCircles(dilation,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=5,minRadius=2,maxRadius=9)

        if circles is not None:
            for circle in circles[0,:]:
                global run_time
                #run_time=run_time+1
                
                g_static_num_x= g_static_num_x +int(circle[0])
                g_static_num_y= g_static_num_y +int(circle[1])
                g_static_num_r= g_static_num_r +int(circle[2])
                #print(run_time)
                #在找球图用指定颜色标记出圆的位置
                #img = cv2.circle(image_find_ball,(temp_ball_x,temp_ball_y),temp_ball_r+2,(0,0,255),-1)
                g_average_mag=g_average_mag+1
        else:
            print("START---------ERROR")
        if g_average_mag >= 10:
            g_start_search_flag = 3
            ball_start_x = int(g_static_num_x/g_average_mag)
            ball_start_y = int(g_static_num_y/g_average_mag)
            ball_r = int(g_static_num_r/g_average_mag)
      
            g_average_mag =   0   
            g_static_num_x= 0
            g_static_num_y= 0
            g_static_num_r= 0
            g_ball_start_flag =1
            print("Ball---------OK")
            
    elif g_start_search_flag == 1:
        '''
        结果处理，主要找出结果坐标

        '''
        image_find_end = image_original[ tablet_y:(tablet_y+tablet_h),tablet_x:(tablet_x+tablet_w)]
        
        lower_blue=np.array([0,43,46])
        upper_blue=np.array([20,255,255])
        
        hsv = cv2.cvtColor(image_find_end, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv,7)
        # get mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        #cv2.imshow('O2', mask)
        # detect blue
        res = cv2.bitwise_and(image_find_end, image_find_end, mask=mask)

        res = cv2.medianBlur(res,3)
  
        res = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
       
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        retval_end, result_end = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY);
        result_end = cv2.medianBlur(result_end,3)
        #cv2.imshow('OK222', result_end)
        image_end_result, contours, hierarchy = cv2.findContours(result_end,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
           
        #temp_end_x, temp_end_y, temp_end_w, temp_end_h = cv2.boundingRect(contours[0])  
        if contours is not None:
            for i in range(0,len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                if w<30 and h<30 and w>5 and h>5:
             
                    g_average_mag=g_average_mag+1     
                    
                    g_static_num_x= g_static_num_x + x
                    g_static_num_y= g_static_num_y + y
                    g_static_num_w= g_static_num_w + w
                    g_static_num_h= g_static_num_h + h
        else:
            print("END---------ERROR")
                
        print("Find")       
        '''
        circles= cv2.HoughCircles(result_end,cv2.HOUGH_GRADIENT,1,100,param1=25,param2=1,minRadius=1,maxRadius=30)
        temp_end_x= 0
        temp_end_y = 0
        temp_end_r=0
        if circles is not None:
            for circle in circles[0,:]:               
                temp_end_x=int(circle[0])
                temp_end_y=int(circle[1])
                temp_end_r=int(circle[2])
                print("search_end:",temp_end_r)
                #在原图用指定颜色标记出圆的位置
        #显示新图像
        '''        
     
        if g_average_mag >= 10:
            g_start_search_flag = 2
            g_ball_end_flag = 1
            ball_end_x = int(g_static_num_x/g_average_mag)
            ball_end_y = int(g_static_num_y/g_average_mag)
            ball_end_w = int(g_static_num_w/g_average_mag)
            ball_end_h = int(g_static_num_h/g_average_mag)
            
            g_average_mag =   0   
            g_static_num_x= 0
            g_static_num_y= 0
            g_static_num_w= 0
            g_static_num_h= 0
            print("END---------OK")
            
      
    elif g_start_search_flag == 4:
        image_find_ball = image_binary[ tablet_y:(tablet_y+tablet_h),tablet_x:(tablet_x+tablet_w)]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7) )
        dilation = cv2.dilate(image_find_ball, kernel,iterations = 2)
       
        #dilation = cv2.erode(dilation, kernel,iterations = 1)
        cv2.imshow("222222",dilation)
        circles= cv2.HoughCircles(dilation,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=3,minRadius=2,maxRadius=9)

        if circles is not None:
            for circle in circles[0,:]:
                global run_time
                global serial
                run_time=run_time+1
                ball_x= int(circle[0])
                ball_y= int(circle[1])
                num_i+=1
                #g_static_num_r= int(circle[2])                
                #fps.update()
                if flag_move == 1 :#and run_time==3:
                    '''
                    run_time = 0
                    move_end_x = 100
                    move_end_y = 100
                    speed_x = ball_x - ball_last_x
                    flag_sign_x = 0
                    if speed_x>=0:
                        flag_sign_x =0 
                    else:
                        flag_sign_x = 1
                    speed_y = ball_y - ball_last_y
                    flag_sign_y = 0
                    if speed_y>=0:
                        flag_sign_y =0 
                    else:
                        flag_sign_y = 1
                    error_x = ball_x - move_end_x
                    flag_sign_error_x = 0
                    if error_x>=0:
                        flag_sign_error_x =0 
                    else:
                        flag_sign_error_x = 1
                    error_y = ball_y - move_end_y
                    flag_sign_error_y = 0
                    if error_y>=0:
                        flag_sign_error_y =0 
                    else:
                        flag_sign_error_y = 1  
                    print(  255,254,ball_x ,ball_y,move_end_x,move_end_x,
                                          flag_sign_y,abs(speed_y),
                                          flag_sign_error_x,abs(error_x),
                                          flag_sign_error_y,abs(error_y),253,252   )
                    '''
                    if (map_path[ball_move_i][0] - ball_x)<7and (map_path[ball_move_i][0] - ball_x)>-7:
                        ball_move_x_ok = 1
                        map_path[ball_move_i][0] = ball_x
                    if (map_path[ball_move_i][1] - ball_y)<7and (map_path[ball_move_i][1] - ball_y)>-7:
                        ball_move_y_ok = 1
                        map_path[ball_move_i][1] = ball_y
                    if ball_move_x_ok == 1 and ball_move_y_ok == 1   :
                        ball_move_y_ok = 0
                        ball_move_x_ok = 0
                        ball_move_i+=1
                    #print(map_path[i][0],map_path[i][1])
                    Uart_buf = bytearray([255,254,ball_x ,ball_y,int(map_path[ball_move_i][0]),int(map_path[ball_move_i][1]),
                                          0,0,
                                          0,0,
                                          0,0,253,252])
                    '''print(  255,254,ball_x ,ball_y,int(map_path[ball_move_i][0]),int(map_path[ball_move_i][1]),
                                          int(ball_move_i),0,
                                          0,0,
                                          0,0,253,252   )'''
                    if num_i>50:
                        num_i=0
                        print( ball_x ,ball_y)
                    ball_last_x = ball_x
                    ball_last_y = ball_y
                    port.write(Uart_buf)
	
                #ps.stop()
                    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                #run_time = 0                    
                    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    elif g_start_search_flag == 3:
        image_find_path = copy.deepcopy(image_binary)
        
        #构造一个3×3的结构元素   
        element = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))  
        dilate = cv2.dilate(image_find_path, element)  
        erode = cv2.erode(image_find_path, element)  

        #将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像  
        result = cv2.absdiff(dilate,erode);    
        result = cv2.medianBlur(result,5)
        #上面得到的结果是灰度图，将其二值化以便更清楚的观察结果  
        retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY);
        #反色，即对二值图每个像素取反  
        result = cv2.bitwise_not(result);
        resultOK = copy.deepcopy(result)
        imag, contours, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    
                #cv2.rectangle(result, (temp_x,temp_y), (temp_x+temp_w,temp_y+temp_h), (255,255,255), 5)
        #cv2.rectangle(imag, (50,50), (x3+w,y3+h), (255,255,255), 5)
        #提取出平板大小
        image_tablet = resultOK[ tablet_y:(tablet_y+tablet_h),tablet_x:(tablet_x+tablet_w)]
  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
        erosion = cv2.erode(image_tablet,kernel,iterations = 1) 
        dilation = cv2.dilate(erosion, kernel,iterations = 2)
        image_tablet = cv2.medianBlur(dilation,3)
 
        cv2.imshow('end_map', image_tablet)
        g_start_map_flag = 1
        g_start_search_flag = 4
        
        
   
    if g_ball_end_flag == 1:   
        cv2.rectangle(image_original,(int(tablet_x+ball_end_x-1),int(tablet_y+ball_end_y-1)),\
                    (int(tablet_x+ball_end_x+ball_end_w+1),int(tablet_y+ball_end_y+ball_end_h+1)), (0,255,0), 1)
        cv2.rectangle(img_path,(int(tablet_x+ball_end_x-1),int(tablet_y+ball_end_y-1)),\
                    (int(tablet_x+ball_end_x+ball_end_w+1),int(tablet_y+ball_end_y+ball_end_h+1)), (0,255,0), 1)
       
        print("ball_end_x:",ball_end_x)
        print("ball_end_y:",ball_end_y)
        print("ball_end_w:",ball_end_w)
        print("ball_end_h:",ball_end_h)
        #cv2.imshow('ball_end', image_original)
        g_ball_end_flag = 0
    if g_ball_start_flag == 1 :
        cv2.circle(img_path,(int(tablet_x+ball_start_x),int(tablet_y+ball_start_y)),int(ball_r+8),(255,0,0),1)
        cv2.circle(image_original,(int(tablet_x+ball_start_x),int(tablet_y+ball_start_y)),int(ball_r+8),(255,0,0),1)
       
        g_ball_start_flag = 0
        print("ball_start_x:",ball_start_x)
        print("ball_start_y:",ball_start_y)
        print("ball_r:",ball_r)
        #cv2.imshow('ball_start', image_original)
    if g_start_search_flag == 4 :
        #print(ball_x,ball_y)
        cv2.circle(image_original,(int(tablet_x+ball_x),int(tablet_y+ball_y)),int(ball_r+8),(255,0,0),1)
    cv2.circle(image_original,(int(tablet_x+ball_x),int(tablet_y+ball_y)),int(ball_r+8),(255,0,0),1)
    
    cv2.imshow('image_original', image_original)
    '''
    print("tablet_x:",tablet_x)
    print("tablet_y:",tablet_y)
    print("tablet_w:",tablet_w)
    print("tablet_h",tablet_h)
    print("ball_start_x:",ball_start_x)
    print("ball_start_y:",ball_start_y)
    '''
    #迷宫提取完成

    '''
    处理地图坐标

    '''
    if g_start_map_flag == 1:
        g_start_map_flag = 0
        kernel = np.ones((7,7),np.uint8)
        size = image_tablet.shape  
        ratio_x = size[0]/(map_w)
        ratio_y = size[1]/(map_h)
        image_res=cv2.resize(image_tablet,(map_w,map_h),interpolation=cv2.INTER_AREA)
        path_mag = cv2.medianBlur(image_tablet,3)
        #data= copy.deepcopy(res) 

        for i in range(map_w):
            for j in range(map_h):
                if image_res[i][j] < 200 :
                    image_res[i][j] = 0
                else:
                    image_res[i][j] = 255
                
        for i in range(map_w+2):
            for j in range(map_h+2):
                if i==0:
                    map_data_mag[i][j] = '#'
                elif i==map_w+1:
                    map_data_mag[i][j] = '#'
                else:
                    if j==0 or j==map_h+1:
                        map_data_mag[i][j]='#'
                    else:             
                        if image_res[i-1][j-1] == 0:
                            map_data_mag[i][j] = '#'
                        else:
                            map_data_mag[i][j] = '.'
        #cv2.imshow("image",image)
    #cv2.imshow("result",result);

#########################################################
class Node_Elem:
    """
    开放列表和关闭列表的元素类型，parent用来在成功的时候回溯路径
    """
    def __init__(self, parent, x, y, dist):
        self.parent = parent
        self.x = x
        self.y = y
        self.dist = dist
        
        
class A_Star:
    """
    A星算法实现类
    """
    #注意w,h两个参数，如果你修改了地图，需要传入一个正确值或者修改这里的默认参数
    def __init__(self, s_x, s_y, e_x, e_y, w=map_w+2, h=map_h+2):
        self.s_x = s_x
        self.s_y = s_y
        self.e_x = e_x
        self.e_y = e_y
        
        self.width = w
        self.height = h
        
        self.open = []
        self.close = []
        self.path = []
        
    #查找路径的入口函数
    def find_path(self):
        #构建开始节点
        p = Node_Elem(None, self.s_x, self.s_y, 0.0)
        while True:
            #扩展F值最小的节点
            self.extend_round(p)
            #如果开放列表为空，则不存在路径，返回
            if not self.open:
                return
            #获取F值最小的节点
            idx, p = self.get_best()
            #找到路径，生成路径，返回
            if self.is_target(p):
                self.make_path(p)
                return
            #把此节点压入关闭列表，并从开放列表里删除
            self.close.append(p)
            del self.open[idx]
            
    def make_path(self,p):
        global path_len
        path_len = 0
        #从结束点回溯到开始点，开始点的parent == None
        while p:
           
            self.path.append((p.x, p.y))
            # 结果打印，在此处，回溯的
            map_path[path_len][0]=p.x
            map_path[path_len][1]=p.y
            path_len+=1
            #print("self.s_x=%d,self.s_y=%d",p.x,p.y)
            p = p.parent
        
    def is_target(self, i):
        return i.x == self.e_x and i.y == self.e_y
        
    def get_best(self):
        best = None
        bv = 100000000 #如果你修改的地图很大，可能需要修改这个值
        bi = -1
        for idx, i in enumerate(self.open):
            value = self.get_dist(i)#获取F值
            if value < bv:#比以前的更好，即F值更小
                best = i
                bv = value
                bi = idx
        return bi, best
        
    def get_dist(self, i):
        # F = G + H
        # G 为已经走过的路径长度， H为估计还要走多远
        # 这个公式就是A*算法的精华了。
        return i.dist + math.sqrt(
            (self.e_x-i.x)*(self.e_x-i.x)
            + (self.e_y-i.y)*(self.e_y-i.y))*1.2
        
    def extend_round(self, p):
        #可以从8个方向走
        xs = (-1, 0, 1, -1, 1, -1, 0, 1)
        ys = (-1,-1,-1,  0, 0,  1, 1, 1)
        #只能走上下左右四个方向
#        xs = (0, -1, 1, 0)
#        ys = (-1, 0, 0, 1)
        for x, y in zip(xs, ys):
            new_x, new_y = x + p.x, y + p.y
            #无效或者不可行走区域，则勿略
            if not self.is_valid_coord(new_x, new_y):
                continue
            #构造新的节点
            node = Node_Elem(p, new_x, new_y, p.dist+self.get_cost(
                        p.x, p.y, new_x, new_y))
            #新节点在关闭列表，则忽略
            if self.node_in_close(node):
                continue
            i = self.node_in_open(node)
            if i != -1:
                #新节点在开放列表
                if self.open[i].dist > node.dist:
                    #现在的路径到比以前到这个节点的路径更好~
                    #则使用现在的路径
                    self.open[i].parent = p
                    self.open[i].dist = node.dist
                continue
            self.open.append(node)
            
    def get_cost(self, x1, y1, x2, y2):
        """
        上下左右直走，代价为1.0，斜走，代价为1.4
        """
        if x1 == x2 or y1 == y2:
            return 1.0
        return 100000
        
    def node_in_close(self, node):
        for i in self.close:
            if node.x == i.x and node.y == i.y:
                return True
        return False
        
    def node_in_open(self, node):
        for i, n in enumerate(self.open):
            if node.x == n.x and node.y == n.y:
                return i
        return -1
        
    def is_valid_coord(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return test_map[y][x] != '#'
    
    def get_searched(self):
        l = []
        for i in self.open:
            l.append((i.x, i.y))
        for i in self.close:
            l.append((i.x, i.y))
        return l
        
#########################################################
def print_test_map():
    """
    打印搜索后的地图
    """
    for line in test_map:
        print(''.join(line))

def get_start_XY():
    return get_symbol_XY('S')
    
def get_end_XY():
    return get_symbol_XY('E')
    
def get_symbol_XY(s):
    for y, line in enumerate(test_map):
        try:
            x = line.index(s)
        except:
            continue
        else:
            break
    return x, y
        
#########################################################
def mark_path(l):
    mark_symbol(l, '*')
    
def mark_searched(l):
    mark_symbol(l, ' ')
    
def mark_symbol(l, s):
    for x, y in l:
        test_map[y][x] = s
    
def mark_start_end(s_x, s_y, e_x, e_y):
    test_map[s_y][s_x] = 'S'
    test_map[e_y][e_x] = 'E'
    
def tm_to_test_map():
    for line in tm:
        test_map.append(list(line))

    for line in tm:
        map_data_mag.append(list(line))
        
    for line in tm:
        Line.append(list(line))
def find_path():
    s_x, s_y = get_start_XY()
    e_x, e_y = get_end_XY()
    a_star = A_Star(s_x, s_y, e_x, e_y)
    a_star.find_path()
    searched = a_star.get_searched()
    path = a_star.path
    #标记已搜索区域
    mark_searched(searched)
    #标记路径
    mark_path(path)    
    print( "path length is %d",(len(path)))
    print( "searched squares count is %d",(len(searched)))
    #标记开始、结束点
    mark_start_end(s_x, s_y, e_x, e_y)

if __name__ == "__main__":

    #port= serial.Serial('/dev/ttyAMA0',115200)
    port.isOpen()
    print("start")
    tm_to_test_map()
    while True:    
        
        while True:
            find_curve()
            cv2.waitKey(1) 
            if g_start_search_flag == 4:
                break;   
            #key = cv2.waitKey(1) & 0xFF
            #if key == ord("q"):
            #   break; 
        for i in range(map_w+2):
            for j in range(map_h+2):
                #print(map_data_mag[i][j],end='')                 
                test_map[i][j] = map_data_mag[i][j]
            #print(" ")  

        start_x_ = int(ball_start_x/ratio_x)
        start_y_ = int(ball_start_y/ratio_y)
            
     
        end_x_ = int((ball_end_x+ball_end_w/2)/ratio_x)
        end_y_ = int((ball_end_y+ball_end_h/2)/ratio_y)
      
        print("data")
      
        print("tablet_x",tablet_x)
        print("tablet_y",tablet_y)
        print("ball_start_x",ball_start_x)
        print("ball_start_y",ball_start_y)
        print("start_x_",start_x_)
        print("start_y_",start_y_)
        print("end_x_",end_x_)
        print("end_y_",end_y_)
        print("ratio_x",ratio_x)
        print("ratio_y",ratio_y)
        
        Uart_buf = bytearray([255,249,ball_start_x,ball_start_y,
                              int(ball_end_x),int(ball_end_y),0,0,0,0,0,0,248,247])            
        port.write(Uart_buf)
        
                     
                        
        mark_start_end(start_x_,start_y_,end_x_,end_y_)
        #把字符串转成列表    
        find_path()
        print_test_map()
        print(path_len)
        port.write(Uart_buf)

        #global image11
        #global temp_x, temp_y, temp_w, temp_h
        #cv2.line(img,(0,0),(511,511),255,5)

        temp_value = 0
        for i in range(int(path_len/2)):
            temp_value = map_path[i][0]
            map_path[i][0] = map_path[path_len-i-1][0]
            map_path[path_len-i][0] = temp_value
            temp_value = map_path[i][1]
            map_path[i][1] = map_path[path_len-i-1][1]
            map_path[path_len-i][1] = temp_value
        img2 =img_path.copy()
        for i in range(path_len-1):
            cv2.line(img_path,(int(tablet_x+(map_path[i][0]-0)*ratio_x),int(tablet_y+(map_path[i][1]-0)*ratio_y)),\
                     (int(tablet_x+(map_path[i+1][0]-0)*ratio_x),int(tablet_y+(map_path[i+1][1]-0)*ratio_y)),255,3)
            
        for i in range(path_len):
            #map_path[i][0] = int(( map_path[i][0]-0)*ratio_x)
            #map_path[i][1] =int(( map_path[i][1]-0)*ratio_y)
            print(map_path[i][0],map_path[i][1])
            #print(map_path[i][0],map_path[i][1])
        ball_move_i = 0
        result= map_path.copy()
        mag = map_path.copy()
        mag_r = map_path.copy()
        #map_path_piont = map_path.copy()
        start_pos = 0
        start_last_pos = 0
        end_pos = 0
        temp_num = 0
        max_temp_num = 0
        temp_value = 0
        pos_num = 0
        flag_up = 0
        flag_mag = 0
        point_i = int(0)
        for i in range(0,path_len-1):
            if flag_up ==0:
                if mag[i][0] != mag[i+1][0]:
                    for j in range(i,path_len-1):
                        if result[j][1] != result[j+1][1] :
                            if j-i>=2:
                                start_last_pos = j
                                end_pos = i
                                flag_mag = 1
                                max_temp_num = 0
                                flag_up = 1
                                temp_num= 0
                            break
                else:
                    pos_num = pos_num+1
                if i==path_len-2:
                    flag_mag =1
                    end_pos = path_len-1
                    max_temp_num = 0
                    temp_num = 0
                    #print(start_x_pos,end_x_pos,temp_v_value)
                if  flag_mag ==1 and pos_num>2:
                    flag_mag = 0
                    pos_num = 0
                    for j in range(start_pos,end_pos):
                        #print(mag[j][0],temp_num)
                        if mag[j][0] == mag[j+1][0]:
                            temp_num=temp_num+1
                            if temp_num>max_temp_num:
                                max_temp_num = temp_num
                                temp_value =  mag[j][0]
                        else:
                            temp_num = 0
                    map_path_piont[point_i][0] = mag[start_pos][0]
                    map_path_piont[point_i][1] =  mag[start_pos][1]
                    map_path_piont[point_i+1][0] =mag[end_pos][0]
                    map_path_piont[point_i+1][1] =mag[end_pos][1]
                    point_i=point_i+2
                    
                    #print(int(start_pos*ratio_x),int(end_pos*ratio_y),start_last_pos,temp_value,max_temp_num)
                    print(int(mag[start_pos][0]*ratio_x),int(mag[start_pos][1]*ratio_y),\
                          int(mag[end_pos][0]*ratio_x),int(mag[end_pos][1]*ratio_y))
                    
                    cv2.circle(img2,(int(tablet_x+mag[start_pos][0]*ratio_x),int(tablet_y+mag[start_pos][1]*ratio_y)),int(5),(255,0,255),1)
                    cv2.circle(img2,(int(tablet_x+mag[end_pos][0]*ratio_x),int(tablet_y+mag[end_pos][1]*ratio_y)),int(5),(255,0,255),1)
                    
                    for j in range(start_pos,end_pos):
                        mag[j][0] = temp_value
                        mag[ len(mag)-1][0]= temp_value
                    start_pos = start_last_pos
            else:
               if i == start_last_pos:
                    flag_up = 0
        print("gaidian",point_i)
        for i in range(point_i):
            print(map_path_piont[i][0],map_path_piont[i][1])
            Uart_buf = bytearray([255,247,int(map_path_piont[i][0]*ratio_x),int(map_path_piont[i][1]*ratio_y),
                              int(ball_end_x),int(ball_end_y),0,0,0,0,0,0,243,244])            
            port.write(Uart_buf)
            time.sleep(0.01)
        '''start_pos = 0
        start_last_pos = 0
        end_pos = 0
        temp_num = 0
        max_temp_num = 0
        temp_value = 0
        pos_num = 0
        flag_up = 0
        flag_mag = 0
        for i in range(0,path_len-1):
            if flag_up ==0:
                if mag[i][1] != mag[i+1][1]:
                    for j in range(i,path_len-1):
                        if result[j][0] != result[j+1][0] :
                            if j-i>=2:
                                start_last_pos = j
                                end_pos = i
                                flag_mag = 1
                                max_temp_num = 0
                                flag_up = 1
                                temp_num= 0
                            break
                else:
                    pos_num = pos_num+1
                if i==path_len-2:
                    flag_mag =1
                    end_pos = path_len-1
                    max_temp_num = 0
                    temp_num = 0
                    #print(start_pos,end_x_pos,temp_v_value)
                if  flag_mag ==1 and pos_num>2:
                    flag_mag = 0
                    pos_num = 0
                    for j in range(start_pos,end_pos):
                        #print(mag[j][1],temp_num)
                        if mag[j][1] == mag[j+1][1]:
                            temp_num=temp_num+1
                            if temp_num>max_temp_num:
                                max_temp_num = temp_num
                                temp_value =  mag[j][1]
                        else:
                            temp_num = 0
                    #print(start_pos,end_pos,start_last_pos,temp_value,max_temp_num)
                    for j in range(start_pos,end_pos):
                        mag[j][1] = temp_value
                        mag[ len(mag)-1][1]= temp_value
                    start_pos = start_last_pos
            else:
               if i == start_last_pos:
                    flag_up = 0
        '''
        #img2 =img_path.copy()
        for i in range(path_len-1):
            cv2.line(img2,(int(tablet_x+(mag[i][0]-0)*ratio_x),int(tablet_y+(mag[i][1]-0)*ratio_y)),\
                     (int(tablet_x+(mag[i+1][0]-0)*ratio_x),int(tablet_y+(mag[i+1][1]-0)*ratio_y)),255,3)
           

        for i in range(path_len):
            map_path[i][0] = int(( map_path[i][0]-0)*ratio_x)
            map_path[i][1] =int(( map_path[i][1]-0)*ratio_y)   
        #cv2.rectangle(img_path,(int(tablet_x+ball_end_x-1),int(tablet_y+ball_end_y-1)),\
        #            (int(tablet_x+ball_end_x+ball_end_w+1),int(tablet_y+ball_end_y+ball_end_h+1)), (0,0,0), -1)
        #cv2.circle(img_path,(int(tablet_x+ball_start_x),int(tablet_y+ball_start_y)),int(ball_r+8,),(0,0,0),-1)
      
        #cv2.circle(image_end,(int(tablet_x+start_x_*ratio_x),int(tablet_y+start_y_*ratio_y)),10,(255,0,0),1)
        #cv2.circle(image_end,(int(tablet_x+end_x_*ratio_x),int(tablet_y+end_y_*ratio_y)),10,(255,0,0),1)
        cv2.imshow("path",img_path)
        cv2.imshow("path2",img2)
        port.write(Uart_buf)  
        flag_move = 1
        while True:
            find_curve()
            port_data =port.read(1)
            #print(port_data)
            if operator.eq(port_data,b'1') == True:
                print("OK")
                #g_start_search_flag = 0
                #break;
            #serial.write(ball_y)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break;





