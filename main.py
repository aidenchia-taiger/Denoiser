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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="path to input image", default="test.png")
	parser.add_argument('--d', help="display denoised image", action='store_true')
	parser.add_argument('--o', help="save binarized image", default=None)
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