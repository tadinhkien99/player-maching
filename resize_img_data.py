#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    resize_img_data.py
# @Author:      Kuro
# @Time:        8/27/2021 11:51 PM


import cv2
import glob
import os
import splitfolders
splitfolders.ratio(r'C:\Users\Kuro\Downloads\data', output=r"C:\Users\Kuro\Downloads\output", seed=1337, ratio=(.8, 0.1,0.1))
#
# path = glob.glob('red/*')
# count = 0
# for file in path:
#     print(count)
#     img = cv2.imread(file)
#     img = cv2.resize(img, (96,96))
#     cv2.imwrite('2/' + str(count) + ".jpg",img)
#     count+=1