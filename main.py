import cv2
import numpy as np
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import math
import pdb
from Utils import display, save
from Denoiser import Denoiser

'TODO: Convert userconfig.txt file to json file'

def main():
	"""
	This Denoiser is made up of many modules that make use of OpenCV morphological operations in order to denoise a document image
	as best as possible as a preprocessing step before feeding the doc image into an OCR engine like Tesseract / ABBYY.

	This denoiser can be used in two ways:
	
	1) The user can specify the exact operations to apply as well as the values of the parameters of individual operations 
	   in the userconfig.txt file. For example, in the userconfig.txt file, the first line is 'CROPTEXT T 0.35 8 8'. This means
	   that the croptext

	2) Otherwise, the denoiser can automatically figure out what denoising operations to apply on the image by using some
	   simple rules to estimate the amount of noise in the document image.

	By specifying the --d command line argument, this main() function will first display a window showing the before and after
	effect of applying the 'automatically' denoised output. It will then open up another window showing the before and after
	effect of applying the 'user' denoised output.

	By specifying the --o command line argument, the denoised images will be written to ../out.

	Example usage:
		python3 main.py --i imgs/guanSoon.png --d --o
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="path to input image", default="test.png")
	parser.add_argument('--d', help="display denoised image", action='store_true')
	parser.add_argument('--o', help="save denoised image", default=None)
	args = parser.parse_args()

	img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
	denoiser = Denoiser()
	auto_denoised = denoiser.denoise(args.i, userconfig=False)
	user_denoised = denoiser.denoise(args.i, userconfig=True)

	if args.d:
		displayImages(img, auto_denoised, True)
		displayImages(img, user_denoised, True)

	if args.o:
		cv2.imwrite('../out/{}_auto.png'.format(args.o), auto_denoised)
		cv2.imwrite('../out/{}_user.png'.format(args.o), user_denoised)

	print(np.array_equal(auto_denoised, user_denoised))

def displayImages(img, denoised, hist=False):
	plt.rcParams["figure.figsize"] = [12, 9]
	
	if hist:
		fig, axes = plt.subplots(2,2, tight_layout=True)
		ax0, ax1, ax2, ax3 = axes.flatten()
		ax0.imshow(img, 'gray')
		ax1.imshow(denoised, 'gray')
		ax2.hist(img.ravel(), 256, [0,256])
		ax3.hist(denoised.ravel(), 256, [0,256])

	else:
		fig, axes = plt.subplots(1,2, tight_layout=True)
		ax0, ax1 = axes.flatten()
		ax0.imshow(img, 'gray')
		ax1.imshow(denoised, 'gray')

	plt.show()


if __name__ == '__main__':
	main()