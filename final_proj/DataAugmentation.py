import numpy as np
import glob
import pandas as pd
import cv2
import random
import os
os.environ['DISPLAY'] = ':0'


class utils_augmentation:
    def horizontal_flip(self, img):
        return cv2.flip(img, -1)


    def vertical_flip(self, img):
        return cv2.flip(img, 1)


    def rnd_rotatation(self, img):
        h, w, _ = img.shape
        degree = random.randint(-45,45)
        M = cv2.getRotationMatrix2D((w/2,h/2), degree, 1)
        return cv2.warpAffine(img, M, (w, h))


    def rnd_crop(self, img):
        x = random.randint(0,56)
        y = random.randint(0,56)
        img = img[x : x + 200, y : y + 200]
        return cv2.resize(img, (256,256))


    def invert_color(self, img): # invert the pixel intensities
        return cv2.bitwise_not(img)


    def change_brightness(self, img, factor=1.5): 
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
        return cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)


    def change_hue(self, img, saturation = 180):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        new_img = img[:, :, 2]
        new_img = np.where(new_img <= 255 + saturation, new_img - saturation, 255)
        img[:, :, 2] = new_img
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


    def add_guassian_noise(self, img):
        h,s,v = cv2.split(img)
        s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return cv2.merge([h,s,v])


    def add_saltpepper_noise(self, img):
        h, w, _ = img.shape
        number_of_pixels = random.randint(9000, 12000)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, h - 1)
            x_coord = random.randint(0, w - 1)
            img[y_coord][x_coord] = 255
        number_of_pixels = random.randint(9000, 12000)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, h - 1)
            x_coord = random.randint(0, w - 1)
            img[y_coord][x_coord] = 0
        return img


    def ResizeImage(self, img):
        h, w, d = img.shape
        dim = 256
        img = cv2.resize(img,(dim,dim))
        return img


    def RGBcentered(self, path):
        images = glob.glob(path + '/*.jpg')
        assert images
        output_path = './bookRGB/'
        lable = 'book'

        # Calculate the mean value of each RGB channel in the entire dataset
        averageRGB = [0,0,0]
        for fname in sorted(images):
            img = cv2.imread(fname)
            # image_averageRGB = np.mean(img, axis = 0)
            # tot_averageRGB += np.mean(averageRGB_row, axis = 0)
            averageRGB += np.mean(np.mean(img, axis = 0), axis = 0)
        
        averageRGB = averageRGB.astype(np.int)
        
        # Subtract the mean to each pixel of each image
        num = 1
        for fname in sorted(images):
            img = cv2.imread(fname)
            img = np.asarray(img, np.int)
            img -= averageRGB
            cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", img)
            num += 1


    def PreProcess(self, img):
        img = self.ResizeImage(img)
        # img = RGBcentered(img)
        augmented = []
        augmented.append(img)
        augmented.append(self.horizontal_flip(img))
        augmented.append(self.vertical_flip(img))
        augmented.append(self.rnd_rotatation(augmented[0]))
        augmented.append(self.rnd_rotatation(augmented[1]))
        augmented.append(self.rnd_rotatation(augmented[2]))
        size = len(augmented)
        i = 0
        while i < 2:
            augmented.append(self.invert_color(augmented[random.randint(0, size-1)]))
            augmented.append(self.change_brightness(augmented[random.randint(0, size-1)]))    
            augmented.append(self.change_hue(augmented[random.randint(0, size-1)]))    
            augmented.append(self.add_guassian_noise(augmented[random.randint(0, size-1)]))    
            augmented.append(self.add_saltpepper_noise(augmented[random.randint(0, size-1)])) 
            i += 1  
        return augmented


class PreProcessor(utils_augmentation):
    def __init__(self, obj_names, input_folder, output_folder):
        self.data_augmentation(obj_names, input_folder, output_folder)

    def data_augmentation(self, obj_names, input_folder, output_folder):

        # input_folder = image folder to process
        # output_folder = folder where to save processed images

        image_augmented = []
        lables = []

        for i in obj_names:
            j = 1 # index for images name
            data_path = input_folder + '/' + str(i)
            filenames = [i for i in os.listdir(data_path)]

            for fname in filenames:
                img = cv2.imread(data_path + '/' + fname)
                augmented = self.PreProcess(img) # preprocess the image: now for every img you get 36 modified copies
                # save new images in a new folder
                directory = "./" + output_folder
                os.chdir(directory)
                for img in augmented:
                    image_augmented.append(img)
                    lables.append(i)
                    # format : 'book0001.jpg' ...
                    cv2.imwrite(str(i) + f"{j:04d}" + ".jpg", img)  
                    j += 1 
                directory = "../"
                os.chdir(directory) 
            
        print(len(image_augmented))
        print(len(lables))

        # # # # Save the dataset
        np.savez("classification/ImageDataset_Augmented", image_augmented)
        np.savez("classification/Lable", lables)
        # # # Load the dataset
        # data = np.load("ImageDataset_Augmented.npz")
        # ImageDataset_Augmented = data['arr_0']
        # # # Load the lable
        # data = np.load("Lable1.npz")
        # Lable = data['arr_0']

    