import cv2
import numpy as np
import os
import argparse
from scipy.ndimage.filters import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
import pdb
from scipy.signal import convolve2d
import math

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="path to input image")
	parser.add_argument('--o', help="save binarized image",default="denoised.png")
	parser.add_argument('--t', help="threshold type: global, adaptive, otsu", default='otsu')
	parser.add_argument('--d', help="apply deshadow", action='store_true')
	args = parser.parse_args()

	#plotHist(args.i)
	
	img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
	#print(estimate_sigma(img, average_sigmas=True, multichannel=True))
	
	if args.d:
		img = deshadow(img)
	
	if args.t == 'global':
		_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

	elif args.t == 'adaptive':
		_, img = cv2.threshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2)

	elif args.t == 'otsu':
		_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

	elif args.t == 'morph':
		#kernel3 = np.ones((2,2),np.uint8)
		#img = cv2.erode(img, kernel3, iterations = 1)
		kernel = np.ones((2,2),np.uint8)
		dilation = cv2.dilate(img,kernel,iterations = 2)
		kernel2 = np.ones((2,2),np.uint8)
		erosion = cv2.erode(dilation, kernel2, iterations = 6)
		img = erosion
	
	
	print(img)

	cv2.imwrite(args.o, img)

	#plt.imshow(img, 'gray')
	#plt.show()


def plotHist(path):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	plt.hist(img.ravel(), 256, [0, 256])
	plt.show()


def check_noise_of_imgs():
	os.chdir('nric/')
	files = os.listdir(os.getcwd())
	for file in files:
		if '.png' in file or '.tif' in file:
			img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
			#print('File {} \t| Sigma {}'.format(file, estimate_sigma(img, average_sigmas=True, multichannel=True)))
			print('File: {} \t| Sigma: {}'.format(file, estimate_noise(img)))

def estimate_noise(img):

  H, W = img.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma


def deshadow(img, maxKernel=10, medianKernel=17):

    bg_img = maximum_filter(img, size =(maxKernel,maxKernel)) # Max Filter

    bg_img = cv2.medianBlur(bg_img, medianKernel) # Median Filter

    diff_img = 255 - cv2.absdiff(img, bg_img) # Extract foreground

    norm_img = np.empty(diff_img.shape)
    norm_img = cv2.normalize(diff_img, dst=norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) # Normalize pixels
    
    #plt.imshow(norm_img,'gray')
    #plt.show()
    return diff_img


if __name__=="__main__":
	main()
	#check_noise_of_imgs()