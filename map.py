# -*- coding: utf-8 -*-
import numpy as np
import heapq as hq
import matplotlib.pyplot as plt
import os

from time import sleep

from tensorflow.python.keras.models import Model, load_model

import sys
import signal
import serial
from picamera import Color, PiCamera
from PIL import Image

MODEL_NAME = 'rubbish.hd5'

# 照片存放的
IMAGE_PATH = '/home/pi/Desktop/Photos/'
dict = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'wall'}


ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
ser.write("l".encode('utf-8'))
def emergency(signal,frame):
    print("STOP")
    sys.exit(0)

signal.signal(signal.SIGINT, emergency)
model = load_model(MODEL_NAME)
demoCamera = PiCamera()
demoCamera.resolution = (640,480)
demoCamera.start_preview()
demoCamera.annotate_background = Color('white')
demoCamera.annotate_background = Color('red')
demoCamera.annotate_text = "SWS2009B - 2019"
demoCamera.brightness = 50
counter=0

BOX_SIZE = 0.2

N = 0
S = 2
W = 3
E = 1

# 记录之前的行走记录
TURN_LEFT = 4
TURN_RIGHT = 5
GO_AHEAD = 6
route = []
# 各个方向的距离
dist = {'front': 1, 'back': 1, 'left': 1, 'right': 1}
# 前进的向量
pos_vec = {N: (-1, 0), S: (1, 0), W: (0, -1), E: (0, 1)}
# 地图
room_map = None
# 方向
oriented = S
# 位置
pos = np.array([1, 1], int)
# 访问过
visited = {'front': False, 'back': False, 'left': False, 'right': False}
# 坐标转换
ori_dir = {'front': S, 'back': N, 'left': E, 'right': W}
dir_ori = {S: 'front', N: 'back', W: 'right', E: 'left'}
# 计算机视图
view = None
# 记数
count = 0
# records
record = []


class squre:
    def __init__(self, pos, G, H, dir=-1, pi=None):
        self.pos = pos
        self.G = G
        self.H = H
        self.F = G + H
        self.pi = pi
        if dir != -1:
            self.path = pi.path + [dir]
        else:
            self.path = []

    def __lt__(self, other):
        return self.F < other.F


def predict(filename):
    img = Image.open(filename)
    img = img.resize((249, 249))
    imgarray = np.array(img) / 255.0
    imgarray = np.expand_dims(imgarray, axis=0)
    result = model.predict(imgarray)
    final = np.argmax(result)
    prob = result[final]
    return final, prob


def get_from_snaer():
    ser.write('H'.encode('utf-8'))
    response = ser.readline()
    return float(response)


def get_dist():
    global dist
    update_dir()
    check_visited()
    # 前后左右的数据都是从传感器获取的

    # front, back, left, right = get_fake_dist()
    front = get_from_snaer()

    dist['front'] = front
    # dist['back'] = back
    # dist['left'] = left
    # dist['right'] = right

    # Draw the wall
    # for key, value in dist.items():
    #     if value == 0:
    #         view[(pos + pos_vec[ori_dir[key]])[0], (pos + pos_vec[ori_dir[key]])[1]] = 0
    if front == 0:
        view[(pos+pos_vec['front'])[0], (pos+pos_vec['front'])[1]] = 0


def check_visited():
    global visited
    for key, value in ori_dir.items():
        temp = pos + np.array(pos_vec[value])
        visited[key] = bool(room_map[temp[0], temp[1]])

def update_dir():
    global ori_dir
    ori_dir['front'] = oriented
    ori_dir['back'] = (oriented + 2) % 4
    ori_dir['left'] = (oriented + 3) % 4
    ori_dir['right'] = (oriented + 1) % 4

    dir_ori[oriented] = 'front'
    dir_ori[(oriented + 2) % 4] = 'back'
    dir_ori[(oriented + 3) % 4] = 'left'
    dir_ori[(oriented + 1) % 4] = 'right'



def get_fake_dist():
    fake_dist = {N: 0, S: 0, W: 0, E: 0}

    row, col = pos[0], pos[1]
    while room_map[row + 1, col] <= 1:
        fake_dist[S] += 1
        row += 1
    row = pos[0]

    while room_map[row - 1, col] <= 1:
        fake_dist[N] += 1
        row -= 1
    row = pos[0]

    while room_map[row, col - 1] <= 1:
        fake_dist[W] += 1
        col -= 1
    col = pos[1]

    while room_map[row, col + 1] <= 1:
        fake_dist[E] += 1
        col += 1

    front = fake_dist[oriented]
    back = fake_dist[(oriented + 2) % 4]
    left = fake_dist[(oriented + 3) % 4]
    right = fake_dist[(oriented + 1) % 4]

    return front, 2, 2, right


def turn_left():
    ser.write('u'.encode('utf-8'))
    print("左转！！！")
    global oriented, view
    oriented = (oriented + 3) % 4
    route.append(TURN_LEFT)
    get_dist()
    return


def turn_right():
    ser.write('r'.encode('utf-8'))
    print("右转！！！")
    global oriented, view
    oriented = (oriented + 1) % 4
    route.append(TURN_RIGHT)
    get_dist()
    return


def go_ahead(step=1):
    ser.write('f'.encode('utf-8'))
    sleep(2)
    global count
    count += 1
    print("前进！！！")
    global dist, pos, view
    pos = pos + np.array(pos_vec[oriented]) * step
    room_map[pos[0], pos[1]] = 1
    route.append(GO_AHEAD)
    get_dist()
    return


def go_back():
    ser.write('j'.encode('utf-8'))
    print('后退！！！')
    global dist, pos
    pos = pos - np.array(pos_vec[oriented])
    get_dist()
    return


def adjust_to(i):
    if i == (oriented + 1) % 4:
        turn_right()
    elif i == (oriented + 3) % 4:
        turn_left()
    elif i == (oriented + 2) % 4:
        turn_left()
        turn_left()


def a_star(t=np.zeros([3, 3])):
    '''寻路代码'''
    if check() and t.all() != 1:
        return
    global pos, room_map
    close = []
    found = False
    path = []
    if t.all() == 0:
        t = np.argwhere(room_map == 0)[0]
    open = [squre(pos, 0, abs(t[0] - pos[0]) + abs(t[1] - pos[1]))]
    hq.heapify(open)
    while not found:
        cur = hq.heappop(open)
        close.append(list(cur.pos))
        for key, value in pos_vec.items():
            t_pos = cur.pos + value
            if list(t_pos) in close or room_map[t_pos[0], t_pos[1]] == 2:
                continue
            temp = squre(t_pos, cur.G + 1, abs(t[0] - t_pos[0]) + abs(t[1] - t_pos[1]), key, cur)
            if (t_pos==t).all():
                found = True
                path = temp.path
                break
            hq.heappush(open, temp)
    for i in path:
        adjust_to(i)
        go_ahead()

    adjust_to(S)


def car_rtn():
    last_act = route.pop()
    if last_act == TURN_LEFT:
        turn_right()
        route.pop()
    if last_act == TURN_RIGHT:
        turn_left()
        route.pop()
    if last_act == GO_AHEAD:
        go_back()


def occur():
    temp = (oriented+2) % 4
    if oriented == S or oriented == N:
        adjust_to(E)
        if dist['front'] != 0:
            go_ahead()
            adjust_to(temp)
            return
    a_star()


def check():
    if 0 not in room_map:
        return 1


def print_img():
    '''Draw the final map.'''
    plt.matshow(view)
    plt.show()
    print(count)

def init():
    '''初始化参数'''
    global view, room_map
    view[0, :], view[-1, :], view[:, 0], view[:, -1] = 0, 0, 0, 0
    # DEV 人工生成障碍
    room_map[0, :], room_map[-1, :], room_map[:, 0], room_map[:, -1] = 2, 2, 2, 2
    room_map[4:6, 6:10] = 2
    room_map[3:6, 1:3] = 2
    room_map[7:9, 3:8] = 2

    room_map[pos[0], pos[1]] = 1
    get_dist()


def main_loop():
    global counter, record
    while True:
        if visited[dir_ori[W]] == 0 and (oriented == N or oriented == S):
            print("前一行没有走过！！")
            pre_ori = oriented
            adjust_to(W)

            while dist['front'] != 0:
                go_ahead()
            if pre_ori == N or pre_ori == S:
                if dist['front'] == 0 and visited[dir_ori[E]]:
                    car_rtn()
                adjust_to(pre_ori)

        elif dist['front'] != 0:
            go_ahead()
        else:
            print('前方有障碍物！！！')
            SNAPSHOT = 'snapshot' + str(counter)
            file_name = IMAGE_PATH + SNAPSHOT + '.jpg'
            demoCamera.capture(file_name)
            print(SNAPSHOT)
            counter += 1
            final, prob = predict(file_name)
            if final != 5:
                record.append([pos, dict[final]])
            # os.system('ding.wav')
            occur()
            if check():
                print('结束了！回家！')
                a_star(np.array([1, 1], int))
                print_img()
                break


def main():
    global room_map, view, record
    x_width, y_width = 2, 1.6
    room_map = np.zeros([int(y_width / BOX_SIZE) + 2, int(x_width / BOX_SIZE) + 2], int)
    view = np.ones([int(y_width / BOX_SIZE) + 2, int(x_width / BOX_SIZE) + 2], int)
    init()

    main_loop()
    print(record)


if __name__ == '__main__':
    main()
