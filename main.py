import cv2
import numpy as np
import os
import argparse
from scipy.ndimage.filters import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
from scipy.signal import convolve2d
import math
import pdb

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="path to input image", default="guanSoon.png")
	parser.add_argument('--o', help="save binarized image",default="denoised.png")
	args = parser.parse_args()

	img = cv2.imread(args.i)
	#print(estimate_noise(img))
	#img = cropTextBbox(img)
	#pdb.set_trace()
	img = threshold(img, 'global')
	img = morph(img)
	display(img)
	cv2.imwrite(args.o, img)




def display(img, window=''):
	img = cv2.resize(img, (1000, 1000))
	cv2.imshow(window, img)
	cv2.waitKey(0)

def estimate_noise(img):
	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	H, W = img.shape

	M = [[1, -2, 1],
		 [-2, 4, -2],
	     [1, -2, 1]]

	sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
	sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
	return sigma


###############################################
def threshold(img, method='otsu'):
	if method == 'global':
		_, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

	elif method == 'adaptive':
		_, img = cv2.threshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2)

	elif method == 'otsu':
		_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

	return img

###############################################
def morph(img, dKernel=(2,2), eKernel=(1,1)):
	dKernel = np.ones(dKernel, np.uint8)
	dilation = cv2.dilate(img, dKernel,iterations = 2) # makes white grow
	eKernel = np.ones(eKernel, np.uint8)
	erosion = cv2.erode(dilation, eKernel, iterations = 1)	# makes black grow
	return erosion

###############################################
def cropTextBbox(img):
	'Finds the texts in img and returns an image with the texts against a white background'
	rgb = cv2.pyrDown(img)
	small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

	_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
	connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

	# using RETR_EXTERNAL instead of RETR_CCOMP
	contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	#For opencv 3+ comment the previous line and uncomment the following line
	#_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	mask = np.zeros(bw.shape, dtype=np.uint8)
	white = np.zeros_like(rgb)
	white[white==0] = 255

	# If fail to find any contours, return the original image
	print(len(contours))
	if len(contours) == 0:
		return bw

	for idx in range(len(contours)):
	    x, y, w, h = cv2.boundingRect(contours[idx])
	    mask[y:y+h, x:x+w] = 0
	    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
	    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

	    if r > 0.45 and w > 8 and h > 8:
	        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (255, 255, 255), 2)
	        out = rgb[y:y+h, x:x+w]
	        white[y:y+h, x:x+w] = out

	#cv2.imwrite('cropped2.png', white)
	return white

###############################################
def deshadow(img, maxKernel=10, medianKernel=17):

    bg_img = maximum_filter(img, size =(maxKernel,maxKernel)) # Max Filter

    bg_img = cv2.medianBlur(bg_img, medianKernel) # Median Filter

    diff_img = 255 - cv2.absdiff(img, bg_img) # Extract foreground

    norm_img = np.empty(diff_img.shape)
    norm_img = cv2.normalize(diff_img, dst=norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) # Normalize pixels
    
    #display(img)
    return diff_img

if __name__ == '__main__':
	main()