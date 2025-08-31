import csv
import math
import os

import cv
import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import skimage
import skimage.feature as feature
from scipy.stats import skew, kurtosis
from PIL import Image, ImageStat

def getListOfFiles(image_folder):
    listOfFile = os.listdir(image_folder)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(image_folder, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles
class FE:

    def build_filters(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
            return filters

    def process(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
            return accum

    def corner(self, img):
        image = cv2.imread(img)
        operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        operatedImage = np.float32(operatedImage)
        dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
        dest = cv2.dilate(dest, None)
        image[dest > 0.01 * dest.max()] = [0, 0, 255]
        imgval = np.sum(image)
        return imgval

    def Interest_Point(self, imgname):
        img = cv2.imread(imgname)
        image = Image.open(imgname).convert("L")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pixel_values = list(image.getdata())
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        sift_image = cv2.drawKeypoints(gray, keypoints, img)
        flatlist = [data for subimg in sift_image.tolist() for data in subimg]
        smdata = [eelmnt for elst in flatlist for eelmnt in elst]
        return (sum(smdata) / len(pixel_values))

    def Entropy(self, imgname):
        img = cv2.imread(imgname, 0)
        lst = img.tolist()
        Hist = pd.Series(lst)
        probs = Hist.value_counts(normalize=True)
        entropy = -1 * np.sum(np.log2(probs) * probs)
        return entropy

    def Dissimilarity(self, imgname):
        image_spot = cv2.imread(imgname)
        gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
        graycom = skimage.feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                               levels=256)
        dissimilarityvals = skimage.feature.graycoprops(graycom, 'dissimilarity')
        dissimilarity = str(dissimilarityvals).replace(' [', '').replace('[', '').replace(']', '')
        return dissimilarity

    def get_pixel(self, img, center, x, y):
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def ltp_calculated_pixel(self, img, x, y):
        center = img[x][y]
        val_ar = []
        val_ar.append(self.get_pixel(img, center, x - 1, y + 1))  # top_right
        val_ar.append(self.get_pixel(img, center, x, y + 1))  # right
        val_ar.append(self.get_pixel(img, center, x + 1, y + 1))  # bottom_right
        val_ar.append(self.get_pixel(img, center, x + 1, y))  # bottom
        val_ar.append(self.get_pixel(img, center, x + 1, y - 1))  # bottom_left
        val_ar.append(self.get_pixel(img, center, x, y - 1))  # left
        val_ar.append(self.get_pixel(img, center, x - 1, y - 1))  # top_left
        val_ar.append(self.get_pixel(img, center, x - 1, y))  # top
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
        return val

    def HOG(self, image):
        image = cv2.imread(image, 0)
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                                winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        hist = hog.compute(image, winStride, padding, locations)
        return hist

    def Brightness(self, imgname):
        image = Image.open(imgname)
        greyscale_image = image.convert('L')
        histogram = greyscale_image.histogram()
        pixels = sum(histogram)
        brightness = scale = len(histogram)
        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)
        return 1 if brightness == 255 else brightness / scale

    def Blobs(self, imgname):
        image = cv2.imread(imgname)
        size = (640, 720)
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=size, swapRB=True)
        return (np.array(blob).shape)

    def Contrast(self, imgname):
        image_spot = cv2.imread(imgname)
        gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
        graycom = skimage.feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                               levels=256)
        contrastvals = skimage.feature.graycoprops(graycom, 'contrast')
        contrast = str(contrastvals).replace(' [', '').replace('[', '').replace(']', '')
        return contrast

    def Homogeneity(self, imgname):
        image_spot = cv2.imread(imgname)
        gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
        graycom = skimage.feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                               levels=256)
        homogeneityvals = skimage.feature.graycoprops(graycom, 'homogeneity')
        homogeneity = str(homogeneityvals).replace(' [', '').replace('[', '').replace(']', '')
        return homogeneity

    def Energy(self, imgname):
        image_spot = cv2.imread(imgname)
        gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
        graycom = skimage.feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                               levels=256)
        energyvals = skimage.feature.graycoprops(graycom, 'energy')
        energy = str(energyvals).replace(' [', '').replace('[', '').replace(']', '')
        return energy

    def Correlation(self, imgname):
        image_spot = cv2.imread(imgname)
        gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
        graycom = skimage.feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                               levels=256)
        correlationval = skimage.feature.graycoprops(graycom, 'correlation')
        correlation = str(correlationval).replace(' [', '').replace('[', '').replace(']', '')
        return correlation

    # Geometric Transformations
    def shape(self, image):
        img = cv.imread(image)
        height, width = img.shape[:2]
        res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
        resval = np.sum(res)
        return resval

    def Translation(self, image):
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        rows, cols = img.shape
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        dst = cv.warpAffine(img, M, (cols, rows))
        dstval = np.sum(dst)
        return dstval

    def Rotation(self, image):
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        rows, cols = img.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
        rot = cv.warpAffine(img, M, (cols, rows))
        rotval = np.sum(rot)
        return rotval

    gray_level = 17

    def dominant_color(self, img):
        max_gray_level = 0
        (height, width) = img.shape
        for y in range(height):
            for x in range(width):
                if img[y][x] > max_gray_level:
                    max_gray_level = img[y][x]
        return max_gray_level + 1

    def getGlcm(self, input, d_x, d_y):
        srcdata = input.copy()
        ret = [[0.0 for i in range(self.gray_level)] for j in range(self.gray_level)]
        (height, width) = input.shape

        max_gray_level = self.dominant_color(input)
        # If the number of gray levels is greater than gray_level, reduce the gray level of the image to gray_level and reduce the size of the gray level co-occurrence matrix
        if max_gray_level > self.gray_level:
            for j in range(height):
                for i in range(width):
                    srcdata[j][i] = srcdata[j][i] * self.gray_level / max_gray_level
        # Optimum probability
        for j in range(height - d_y):
            for i in range(width - d_x):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0

        for i in range(self.gray_level):
            for j in range(self.gray_level):
                ret[i][j] /= float(height * width)
        return ret

    def LP_Quantization(self, p):
        Con = 0.0
        Eng = 0.0
        Asm = 0.0
        Idm = 0.0
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                Con += (i - j) * (i - j) * p[i][j]
                Asm += p[i][j] * p[i][j]
                Idm += p[i][j] / (1 + (i - j) * (i - j))
                if p[i][j] > 0.0:
                    Eng += p[i][j] * math.log(p[i][j])
        return Asm, Con, -Eng, Idm

    def skewness(self,img):
        im = cv2.imread(img)
        colon_number = 0
        ske = skew(im.reshape(-1, 3)[:, colon_number])
        return ske

    def kurtosis_val(self,img):
        im = cv2.imread(img)
        colon_number = 0
        kurval = kurtosis(im.reshape(-1, 3)[:, colon_number])
        return kurval

    def mean(self,im):
        img = cv2.imread(im)
        img = img[:, :, 0]
        mean = img.mean()
        return (mean)

    def area_value(self,im):
        img = cv2.imread(im)
        img = img[:, :, 1]
        green = img.mean()
        return green

    def size(self,im):
        img = cv2.imread(im)
        img = img[:, :, 2]
        blue = img.mean()
        return blue

    def standard_deviation(self,image):
        im = Image.open(image)
        stat = ImageStat.Stat(im)
        return (sum(stat.stddev))

    def shape(self,image):
        im = cv2.imread(image)
        return(im.shape)



    def dwt(self,image):
        lowpass = np.ones((3, 3)) * (1 / 9)
        highpass_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # sobel filter
        highpass_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # sobel filter
        def conv(image, filter):
            dimx = image.shape[0] - filter.shape[0] + 1
            dimy = image.shape[1] - filter.shape[1] + 1
            ans = np.zeros((dimx, dimy))
            for i in range(dimx):
                for j in range(dimy):
                    ans[i, j] = np.sum(image[i:i + filter.shape[0], j:j + filter.shape[1]] * filter)
            return ans
        l = conv(image, lowpass)
        h = conv(image, highpass_x)
        ll = conv(l, lowpass)  # approximate subband
        lh = conv(l, highpass_x)  # horizontal subband
        hl = conv(l, highpass_y)  # vertical subband
        hh = conv(h, highpass_y)  # diagonal subband
        return ll, lh, hl, hh



