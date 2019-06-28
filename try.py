import cv2
from sklearn.mixture import GaussianMixture
import numpy as np
import pdb

def cluster(img):
	shape = img.shape
	img = img.flatten()
	img = img.reshape(-1, 1)
	gmm = GaussianMixture(n_components=2, verbose=1)
	cluster = gmm.fit_predict(img).reshape(shape)
	pdb.set_trace()

if __name__ == '__main__':
	img = cv2.imread('imgs/marcus_front.jpg', cv2.IMREAD_GRAYSCALE)
	cluster(img)
