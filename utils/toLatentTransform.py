# TODO:consist of random amounts of Gaussianblurring, Gaussian noise, downsampling, partial occlusions and contrast adjustment
import cv2
import numpy as np
import os
import random
import torch
class toLatentTransform:
    def __init__(self, p=0.5):
        """
        初始化toLatentTransform类的实例。

        参数：
        - p：float，表示应用每个噪声函数的概率，默认为0.5。
        """
        self.p = p

    def __call__(self, img):
        """
        将图像转换为潜在空间。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示转换后的图像。
        """
        height, width = img.shape
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if random.random() < self.p:
            img = self.noise1(img)
        if random.random() < self.p:
            img = self.noise2(img)
        if random.random() < self.p:
            img = self.noise3(img)
            
        if random.random() < self.p:
            img = self.noise4(img)
            
        if random.random() < self.p:
            img = self.noise5(img)
        if random.random() < self.p:
            img = self.noise6(img)
        if random.random() < self.p:
            img = self.noise7(img)
        if random.random() < self.p:
            img = self.noise8(img)
        if random.random() < self.p:
            img = self.noise9(img)
        
        img = cv2.resize(img, (width, height))
        return img
    
    def __repr__(self):
        """
        返回toLatentTransform类的字符串表示形式。
        """
        return self.__class__.__name__ + '()'
    
    def noise1(self, img):
        """
        对图像应用高斯模糊。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示应用高斯模糊后的图像。
        """
        kernel_size = random.choice([(3, 3), (5, 5), (7, 7), (9, 9)])
        img = cv2.GaussianBlur(img, kernel_size, random.uniform(0, 5))
        return img
    
    def noise2(self, img):
        """
        对图像添加高斯噪声。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示添加高斯噪声后的图像。
        """
        row, col= img.shape
        gauss = np.random.normal(random.uniform(0, 10), random.uniform(0, 5), (row, col))
        gauss = gauss.astype(np.uint8)
        img = img + gauss
        return img
    
    def noise3(self, img):
        """
        对图像进行降采样。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示进行尺度变换后的图像。
        """
        scale = random.uniform(0.5, 1)
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        return img
    
    def noise4(self, img):
        """
        对图像进行亮度调整。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示进行亮度调整后的图像。
        """
        alpha = random.uniform(0.6, 1)
        img = cv2.convertScaleAbs(img, alpha, beta=0)
        return img
    
    def noise5(self, img):
        """
        在图像上绘制随机直线。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示绘制随机直线后的图像。
        """
        height, width= img.shape
        for _ in range(random.randint(4, 10)):
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)
            thickness = random.randint(3, 5)
            overlay = img.copy()
            cv2.line(overlay, start_point, end_point, color, thickness)
            alpha = random.uniform(0, 1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
    
    def noise6(self, img):
        """
        在图像上绘制随机矩形。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示绘制随机矩形后的图像。
        """
        height, width= img.shape
        recw = round(width / 30)
        rech = round(height / 30)
        for _ in range(random.randint(1, 8)):
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (start_point[0] + recw, start_point[1] + rech)
            color = (0, 0, 0)
            thickness = 5
            overlay = img.copy()
            alpha1 = random.uniform(0, 1)
            alpha2 = random.uniform(0, 1)
            cv2.rectangle(overlay, start_point, end_point, color, -1)
            cv2.addWeighted(overlay, alpha1, img, 1 - alpha1, 0, img)
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (start_point[0] + recw, start_point[1] + rech)
            cv2.rectangle(overlay, start_point, end_point, color, thickness)
            cv2.addWeighted(overlay, alpha2, img, 1 - alpha2, 0, img)
        return img
    
    def noise7(self, img):
        """
        在图像上绘制随机圆形。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示绘制随机圆形后的图像。
        """
        height, width= img.shape
        for _ in range(random.randint(1, 3)):
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(min(width, height) // 6, min(width, height) // 2)
            color = (0, 0, 0)
            thickness = random.randint(3, 5)
            overlay = img.copy()
            alpha =  random.uniform(0, 1)
            cv2.circle(overlay, center, radius, color, thickness)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
    
    def noise8(self, img):
        """
        在图像上绘制随机多边形。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示绘制随机多边形后的图像。
        """
        height, width= img.shape
        for _ in range(random.randint(1, 3)):
            pts = np.array([[random.randint(0, width), random.randint(0, height)] for _ in range(random.randint(3, 6))], np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = (0, 0, 0)
            thickness = random.randint(3, 5)
            overlay = img.copy()
            alpha = random.uniform(0, 1)
            cv2.polylines(overlay, [pts], True, color, thickness)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
    
    def noise9(self, img):
        """
        在图像上绘制随机文本。

        参数：
        - img：numpy.ndarray，表示输入的图像。

        返回：
        - img：numpy.ndarray，表示绘制随机文本后的图像。
        """
        height, width= img.shape
        for _ in range(random.randint(3, 6)):
            fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
            font = random.choice(fonts)
            text = ''
            for _ in range(random.randint(4, 12)):
                text += chr(random.randint(65, 90))
            org = (random.randint(0, width), random.randint(0, height))
            fontScale = random.uniform(0.5, 3)
            color = (0, 0, 0)
            thickness = random.randint(3, 5)
            overlay = img.copy()
            cv2.putText(overlay, text, org, font, fontScale, color, thickness)
            alpha = random.uniform(0, 1)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img
            
        
