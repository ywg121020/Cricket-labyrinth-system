# !/usr/bin/env python 
# -*- coding:utf-8 -*-
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


# port = serial.Serial("/dev/ttyAMA0", baudrate=115200, timeout=1.0)
# Send_data
Uart_buf = [0x55, 0x00, 0xAA, 0x00, 0xAA, 0x00, 0xAA, 0x00, 0xAA, 0x00, 0xAA, 0xAA, 0xAA, 0xAA]
# Setup Usart
port = serial.Serial("/dev/ttyAMA0", baudrate=115200, timeout=0)

tm = [[0 for x in range(map_w + 2)] for y in range(map_h + 2)]


from picMag import picMag
from myMath import find_path,print_test_map,tm_to_test_map,mark_start_end,map_path,path_len,map_path_piont

if __name__ == "__main__":
    picMag.findTablet()

    port.isOpen()

    tm_to_test_map()
    while True:
        picMag.findTablet()
        picMag.findStart()
        picMag.findEnd()
        picMag.getMap()
        picMag.magMap(map_data_mag)

        for i in range(map_w + 2):
            for j in range(map_h + 2):
                # print(map_data_mag[i][j],end='')
                test_map[i][j] = map_data_mag[i][j]

        start_x_ = int(picMag.msg['ball_start_x'] / self.msg['ratio_x'] )
        start_y_ = int(picMag.msg['ball_start_y'] / self.msg['ratio_y'] )

        end_x_ = int((picMag.msg['ball_end_x'] + self.msg['ball_end_w'] / 2) / self.msg['ratio_x'] )
        end_y_ = int((picMag.msg['ball_end_y'] + self.msg['ball_end_h'] / 2) / self.msg['ratio_y'])


        Uart_buf = bytearray([255, 249, self.msg['ball_start_x'], self.msg['ball_start_y'],
                              int(self.msg['ball_end_x']), int(self.msg['ball_end_y']), 0, 0, 0, 0, 0, 0, 248, 247])
        port.write(Uart_buf)

        mark_start_end(start_x_, start_y_, end_x_, end_y_)
        # 把字符串转成列表
        find_path(picMag.map_w,picMag.map_h)
        print_test_map()
        port.write(Uart_buf)

        temp_value = 0
        for i in range(int(path_len / 2)):
            temp_value = map_path[i][0]
            map_path[i][0] = map_path[path_len - i - 1][0]
            map_path[path_len - i][0] = temp_value
            temp_value = map_path[i][1]
            map_path[i][1] = map_path[path_len - i - 1][1]
            map_path[path_len - i][1] = temp_value

        for i in range(path_len):
            # map_path[i][0] = int(( map_path[i][0]-0)*self.msg['ratio_x'] )
            # map_path[i][1] =int(( map_path[i][1]-0)*self.msg['ratio_y'])
            print(map_path[i][0], map_path[i][1])
            # print(map_path[i][0],map_path[i][1])
        ball_move_i = 0
        result = map_path.copy()
        mag = map_path.copy()
        mag_r = map_path.copy()
        # map_path_piont = map_path.copy()
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
        for i in range(0, path_len - 1):
            if flag_up == 0:
                if mag[i][0] != mag[i + 1][0]:
                    for j in range(i, path_len - 1):
                        if result[j][1] != result[j + 1][1]:
                            if j - i >= 2:
                                start_last_pos = j
                                end_pos = i
                                flag_mag = 1
                                max_temp_num = 0
                                flag_up = 1
                                temp_num = 0
                            break
                else:
                    pos_num = pos_num + 1
                if i == path_len - 2:
                    flag_mag = 1
                    end_pos = path_len - 1
                    max_temp_num = 0
                    temp_num = 0
                    # print(start_x_pos,end_x_pos,temp_v_value)
                if flag_mag == 1 and pos_num > 2:
                    flag_mag = 0
                    pos_num = 0
                    for j in range(start_pos, end_pos):
                        # print(mag[j][0],temp_num)
                        if mag[j][0] == mag[j + 1][0]:
                            temp_num = temp_num + 1
                            if temp_num > max_temp_num:
                                max_temp_num = temp_num
                                temp_value = mag[j][0]
                        else:
                            temp_num = 0
                    map_path_piont[point_i][0] = mag[start_pos][0]
                    map_path_piont[point_i][1] = mag[start_pos][1]
                    map_path_piont[point_i + 1][0] = mag[end_pos][0]
                    map_path_piont[point_i + 1][1] = mag[end_pos][1]
                    point_i = point_i + 2

                    # print(int(start_pos*self.msg['ratio_x'] ),int(end_pos*self.msg['ratio_y']),start_last_pos,temp_value,max_temp_num)
                    print(int(mag[start_pos][0] * self.msg['ratio_x'] ), int(mag[start_pos][1] * self.msg['ratio_y']), \
                          int(mag[end_pos][0] * self.msg['ratio_x'] ), int(mag[end_pos][1] * self.msg['ratio_y']))

                    cv2.circle(img2, (
                    int(self.msg['tablet_x'] + mag[start_pos][0] * self.msg['ratio_x'] ), int(self.msg['tablet_y'] + mag[start_pos][1] * self.msg['ratio_y'])), int(5),
                               (255, 0, 255), 1)
                    cv2.circle(img2,
                               (int(self.msg['tablet_x'] + mag[end_pos][0] * self.msg['ratio_x'] ), int(self.msg['tablet_y'] + mag[end_pos][1] * self.msg['ratio_y'])),
                               int(5), (255, 0, 255), 1)

                    for j in range(start_pos, end_pos):
                        mag[j][0] = temp_value
                        mag[len(mag) - 1][0] = temp_value
                    start_pos = start_last_pos
            else:
                if i == start_last_pos:
                    flag_up = 0

        for i in range(point_i):
            print(map_path_piont[i][0], map_path_piont[i][1])
            Uart_buf = bytearray([255, 247, int(map_path_piont[i][0] * self.msg['ratio_x'] ), int(map_path_piont[i][1] * self.msg['ratio_y']),
                                  int(self.msg['ball_end_x']), int(self.msg['ball_end_y']), 0, 0, 0, 0, 0, 0, 243, 244])
            port.write(Uart_buf)
            time.sleep(0.01)

        for i in range(path_len):
            map_path[i][0] = int((map_path[i][0] - 0) * self.msg['ratio_x'] )
            map_path[i][1] = int((map_path[i][1] - 0) * self.msg['ratio_y'])

        port.write(Uart_buf)
        flag_move = 1
        ball_move_i = 0
        ball_move_x_ok = 0
        ball_move_y_ok = 0
        while True:
            ball_x, ball_y, flag = picMag.findBall()
            if flag :
                if (map_path[ball_move_i][0] - ball_x) < 7 and (map_path[ball_move_i][0] - ball_x) > -7:
                    ball_move_x_ok = 1
                    map_path[ball_move_i][0] = ball_x
                if (map_path[ball_move_i][1] - ball_y) < 7 and (map_path[ball_move_i][1] - ball_y) > -7:
                    ball_move_y_ok = 1
                    map_path[ball_move_i][1] = ball_y
                if ball_move_x_ok == 1 and ball_move_y_ok == 1:
                    ball_move_y_ok = 0
                    ball_move_x_ok = 0
                    ball_move_i += 1
                # print(map_path[i][0],map_path[i][1])
                Uart_buf = bytearray(
                    [255, 254, ball_x, ball_y, int(map_path[ball_move_i][0]), int(map_path[ball_move_i][1]),
                     0, 0,
                     0, 0,
                     0, 0, 253, 252])

            port_data = port.read(1)
            # print(port_data)
            if operator.eq(port_data, b'1') == True:
                pass
                # g_start_search_flag = 0
                # break;
            # serial.write(self.msg['ball_end_y'])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break;





