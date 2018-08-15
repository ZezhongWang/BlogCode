# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 17:02:56 2018

@author: ZeZhongWang
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
import random
import time
import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np


cut_width = 91
cut_height = 240
back_width = 480
back_height = 240

class WangYi(object):
    
    def __init__(self):
        self.browser = webdriver.Edge()
        self.back_img = None
        self.cut_img = None
        self.scaling_ratio = 1.0
        
        
    def visit(self, url):
        self.browser.get(url)
        WebDriverWait(self.browser, 10, 0.5).until(EC.element_to_be_clickable((By.CLASS_NAME, 'big-heart')))
        time.sleep(2)
        self.browser.find_element_by_class_name("big-heart").click() 
        
    def get_image(self):
        # 等待加载       
        WebDriverWait(self.browser, 10, 0.5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'yidun_bgimg')))
        back_url= self.browser.find_element_by_class_name("yidun_bg-img").get_attribute('src')
        cut_url = self.browser.find_element_by_class_name("yidun_jigsaw").get_attribute('src')
        # 从url获取图片并保存到本地
        resq = requests.get(back_url)
        file = BytesIO(resq.content)
        back_img = Image.open(file)
        back_img.save("back_img.jpg")
        resq = requests.get(cut_url)
        file = BytesIO(resq.content)
        cut_img = Image.open(file)
        cut_img.save("cut_img.png")
        # opencv读取图片
        self.back_img = cv2.imread("back_img.jpg")
        self.cut_img = cv2.imread("cut_img.png")
        self.scaling_ratio = self.browser.find_element_by_class_name("yidun_bg-img").size['width'] / back_width
        return self.cut_img, self.back_img 
        
    def get_distance(self):
        back_canny = get_back_canny(self.back_img)
        operator = get_operator(self.cut_img)
        pos_x, max_value = best_match(back_canny, operator)
        distance = pos_x * self.scaling_ratio
        return distance
        
    def auto_drag(self, distance):
        element = self.browser.find_element_by_class_name("yidun_slider")
        
        # 这里就是根据移动进行调试，计算出来的位置不是百分百正确的，加上一点偏移
        #distance -= element.size.get('width') / 2
        distance += 13
        has_gone_dist = 0
        remaining_dist = distance
        #distance += randint(-10, 10)
 
        # 按下鼠标左键
        ActionChains(self.browser).click_and_hold(element).perform()
        time.sleep(0.5)
        while remaining_dist > 0:
            ratio = remaining_dist / distance
            if ratio < 0.2:
                # 开始阶段移动较慢
                span = random.randint(5, 8)
            elif ratio > 0.8:
                # 结束阶段移动较慢
                span = random.randint(5, 8)
            else:
                # 中间部分移动快
                span = random.randint(10, 16)
            ActionChains(self.browser).move_by_offset(span, random.randint(-5, 5)).perform()
            remaining_dist -= span
            has_gone_dist += span
            time.sleep(random.randint(5,20)/100)
         
        ActionChains(self.browser).move_by_offset(remaining_dist, random.randint(-5, 5)).perform()
        ActionChains(self.browser).release(on_element=element).perform()
 
        
        
      
def read_img_file(cut_dir, back_dir):
    cut_image = cv2.imread(cut_dir)
    back_image = cv2.imread(back_dir)
    return cut_image, back_image

def best_match(back_canny, operator):
    max_value, pos_x = 0, 0
    for x in range(cut_width, back_width - cut_width):
        block = back_canny[:, x:x + cut_width]
        value = (block * operator).sum()
        if value > max_value:
            max_value = value
            pos_x = x
    return pos_x, max_value
        
def get_back_canny(back_img):
    img_blur = cv2.GaussianBlur(back_img, (3, 3), 0)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 100, 200)
    return img_canny
    
def get_operator(cut_img):
    
    cut_gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)

    _, cut_binary = cv2.threshold(cut_gray, 127, 255, cv2.THRESH_BINARY)
    # 获取边界
    _, contours, hierarchy = cv2.findContours(cut_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 获取最外层边界
    contour = contours[-1]
    # operator矩阵
    operator = np.zeros((cut_height, cut_width))
    # 根据 contour填写operator
    for point in contour:
        operator[point[0][1]][point[0][0]] = 1
    return operator

    
    
if __name__ == '__main__':
    
    page = WangYi()
    page.visit('http://game.academy.163.com/minigame/2018/showcase/detail/143')
    cut_image, back_image = page.get_image()
    distance = page.get_distance()
    page.auto_drag(distance)
    
    
    
    
    
    '''
    browser = webdriver.Chrome()
    browser.get('http://game.academy.163.com/minigame/2018/showcase/detail/125')
    WebDriverWait(browser, 10, 0.5).until(EC.element_to_be_clickable((By.CLASS_NAME, 'big-heart')))
    time.sleep(0.25)
    browser.find_element_by_class_name("big-heart").click()
    
    WebDriverWait(browser, 10, 0.5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'yidun_bgimg')))
    
    back_url= browser.find_element_by_class_name("yidun_bg-img").get_attribute('src')
    cut_url = browser.find_element_by_class_name("yidun_jigsaw").get_attribute('src')
    
    resq = requests.get(back_url)
    file = BytesIO(resq.content)
    back_img = Image.open(file)
    
    resq = requests.get(cut_url)
    file = BytesIO(resq.content)
    cut_img = Image.open(file)
    
    
    
    element = browser.find_element_by_class_name("yidun_slider")
    distance -= element.size.get('width') / 2
    distance += 15
 
    # 按下鼠标左键
    ActionChains(browser).click_and_hold(element).perform()
    time.sleep(0.5)
    while distance > 0:
        if distance > 10:
            # 如果距离大于10，就让他移动快一点
            span = random.randint(5, 8)
        else:
            # 快到缺口了，就移动慢一点
            span = random.randint(2, 3)
        ActionChains(browser).move_by_offset(span, 0).perform()
        distance -= span
        time.sleep(random.randint(10,50)/100)
     
    ActionChains(browser).move_by_offset(distance, 1).perform()
    ActionChains(browser).release(on_element=element).perform()
    '''
