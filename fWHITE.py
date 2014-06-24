#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Some examples of useful code. No, really, i'm not joking
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
import os,sys
import glob
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numpy import *
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T
import cPickle
import time			# What time is it? Adven...
from TheanoNNclassCORE2 import *		# Some cool NN builder here
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
piclist = glob.glob("./GAL/*.jpg")
#piclist = glob.glob("E:\\KAGGLE_G\\images_training_rev1\\100100min\\*.jpg")
i = 0
for file in piclist:
	im = Image.open(file)
	im = im.convert("RGB")
	im = np.array(im)
	for j in xrange(im.shape[2]):
		im[:,:,j] = DataMutate.Normalizer(DataMutate.PCAW(im[:,:,j])).astype('uint8')
	print i
	imsave = Image.fromarray(im)
	imsave = imsave.convert("RGB")
	imsave.save("./GAL/whitened/" + str(i) + ".jpg", "JPEG", quality = 100)
	#imsave.save("E:\\KAGGLE_G\\images_training_rev1\\100100minwhite\\" + str(i) + ".jpg", "JPEG", quality = 100)
	i = i + 1
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#