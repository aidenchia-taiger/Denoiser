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
	parser.add_argument('--i', help="path to input image", default="test2.png")
	parser.add_argument('--o', help="save binarized image", default=None)
	args = parser.parse_args()

	denoiser = Denoiser()
	img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
	denoised = denoiser.denoise(img)
	if denoiser.config['DISPLAY']:
		display(img, denoised, True) if denoiser.config['HIST'] else display(img, denoised, False)

	if args.o:
		cv2.imwrite('out/' + args.o, denoised)

def display(img, denoised, hist=False):
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

class Denoiser:
	def __init__(self):
		self.config = self.read_config(open('config.txt'))

	def denoise(self, img):
		self.cropBackground(img)
		if self.config['CROPTEXT']:
			img = self.cropTextBbox(img)

		if self.config['DESPECKLE']:
			img = self.despeckle(img, self.config['DILATION KERNEL SIZE'], \
									  self.config['DILATION ITERATIONS'], \
									  self.config['EROSION KERNEL SIZE'], \
									  self.config['EROSION ITERATIONS'])



		if self.config['DESHADOW']:
			img = self.deshadow(img, self.config['MAX KERNEL'], self.config['MEDIAN KERNEL'])

		if self.config['BLUR']:
			img = self.blur(img, 'bilateral')

		if self.config['CONTRAST']:
			img = self.increaseContrast(img)

		if self.config['GRADIENT']:
			img = self.gradient(img, self.config['GRADIENT KERNEL SIZE'])

		if self.config['CLOSING']:
			img = self.closing(img, self.config['CLOSING KERNEL SIZE'])

		if self.config['BINARIZE']:
			img = self.binarize(img, self.config['BINARIZATION METHOD'])
		
		return img

	def read_config(self, config):
		dic = {}
		for line in config:
			line = line.strip().split()
			if line[0] == 'DESPECKLE':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['DILATION KERNEL SIZE'] = int(line[2])
				dic['DILATION ITERATIONS'] = int(line[3])
				dic['EROSION KERNEL SIZE'] = int(line[4])
				dic['EROSION ITERATIONS'] = int(line[5])
				self.printIfTrue(line[0], dic, 'DILATION KERNEL SIZE', 'DILATION ITERATIONS', 'EROSION KERNEL SIZE', 'EROSION ITERATIONS')

			elif line[0] == 'BINARIZE':
				dic[line[0]] = True if line[1] == 'T' else False
				if line[2] == str(0): 
					dic['BINARIZATION METHOD'] = 'global'

				elif line[2] == str(1):
					dic['BINARIZATION METHOD'] = 'adaptive'

				else:
					dic['BINARIZATION METHOD'] = 'otsu'
				self.printIfTrue(line[0], dic,'BINARIZATION METHOD')

			elif line[0] == 'DESHADOW':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['MAX KERNEL'] = int(line[2])
				dic['MEDIAN KERNEL'] = int(line[3])
				self.printIfTrue(line[0], dic, 'MAX KERNEL', 'MEDIAN KERNEL')
			
			elif line[0] == 'BLUR':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['BLURRING KERNEL SIZE'] = int(line[2])

				if line[3] == str(0):
					dic['BLURRING METHOD'] = 'average'
				elif line[3] == str(1):
					dic['BLURRING METHOD'] = 'median'
				elif line[3] == str(2):
					dic['BLURRING METHOD'] = 'gaussian'
				elif line[3] == str(3):
					dic['BLURRING METHOD'] = 'bilateral'
				self.printIfTrue(line[0], dic, 'BLURRING METHOD')
			
			elif line[0] == 'DISPLAY':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['HIST'] = True if line[2] == 'T' else False
				self.printIfTrue(line[0], dic, 'HIST')

			elif line[0] == 'GRADIENT':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['GRADIENT KERNEL SIZE'] = int(line[2])
				self.printIfTrue(line[0], dic, 'GRADIENT KERNEL SIZE')

			elif line[0] == 'CLOSING':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['CLOSING KERNEL SIZE'] = int(line[2])
				self.printIfTrue(line[0], dic, 'CLOSING KERNEL SIZE')

			else:
				dic[line[0]] = True if line[1] == 'T' else False
				self.printIfTrue(line[0], dic)
			
		return dic

	def printIfTrue(self, method, dic, *params):
		if dic[method]:
			print('{}: {}'.format(method, dic[method]))
			for param in params:
				print('{}: {}'.format(param, dic[param]))


	def estimate_noise(self, img):
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
	def binarize(self, img, method='otsu', gthreshold=127):
		# adaptive and Otsu requires image to be grayscale
		if method == 'global':
			_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

		elif method == 'adaptive':
			img = cv2.adaptiveThreshold(src=img, dst=img, maxValue=255, \
										adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
								   		thresholdType=cv2.THRESH_BINARY, blockSize=21, C=2)

		elif method == 'otsu':
			_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

		return img

	###############################################
	def cropBackground(self, img):
		imgArea = img.shape[0] * img.shape[1]
		th, threshed = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

		## (2) Morph-op to remove noise
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
		morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

		## (3) Find the max-area contour
		cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
		for idx in range(len(cnt)):
			x,y,w,h = cv2.boundingRect(cnt[idx])
			if w*h / imgArea < 0.1:
				continue
			dst = img[y:y+h, x:x+w]
			cv2.imwrite('crop{}.png'.format(idx), dst)

	###############################################
	def gradient(self, img, kernelSize):
		kernel = np.ones((kernelSize,kernelSize),np.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
		return img

	###############################################
	def closing(self, img, kernelSize):
		kernel = np.ones((kernelSize,kernelSize),np.uint8)
		img =  cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		return img

	###############################################
	def despeckle(self, img, dKernel=2, dIterations=2, eKernel=1, eIterations=1):
		dKernel = np.ones((dKernel, dKernel), np.uint8)
		dilation = cv2.dilate(img, dKernel, iterations = dIterations) # makes white grow
		eKernel = np.ones((eKernel, eKernel), np.uint8)
		erosion = cv2.erode(dilation, eKernel, iterations = eIterations) # makes black grow
		return erosion

	###############################################
	def cropTextBbox(self, img):
		'Finds the texts in img and returns an image with the texts against a white background'
		rgb = cv2.pyrDown(img)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		grad = cv2.morphologyEx(rgb, cv2.MORPH_GRADIENT, kernel)

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
		#print('[INFO] No. of contours found: {}'.format(len(contours)))
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
	def deshadow(self, img, maxKernel=10, medianKernel=17):

	    bg_img = maximum_filter(img, size =(maxKernel,maxKernel)) # Max Filter

	    bg_img = cv2.medianBlur(bg_img, medianKernel) # Median Filter

	    diff_img = 255 - cv2.absdiff(img, bg_img) # Extract foreground

	    norm_img = np.empty(diff_img.shape)
	    norm_img = cv2.normalize(diff_img, dst=norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) # Normalize pixels
	    
	    #display(img)
	    return diff_img

	###############################################
	def blur(self, img, method, kernelSize=3):
		if method == 'average':
			img = cv2.blur(img, (kernelSize, kernelSize))

		elif method == 'median':
			img = cv2.medianBlur(img, kernelSize)

		elif method == 'gaussian':
			img = cv2.GaussianBlur(img, (kernelSize, kernelSize), 0)

		elif method == 'bilateral':
			img = cv2.bilateralFilter(img, 9, 150, 150)

		return img
	###############################################
	def increaseContrast(self, img):
		return cv2.equalizeHist(img)

if __name__ == '__main__':
	main()