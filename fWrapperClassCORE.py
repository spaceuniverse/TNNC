#---------------------------------------------------------------------#
# External libraries
#---------------------------------------------------------------------#
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numpy import *
from numpy import dot, sqrt, diag
from numpy.linalg import eigh
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T
import cPickle
import time					# What time is it? Adven...
import matplotlib.pyplot as plt
from fTheanoNNclassCORE import *				# Some cool NN builder here
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
class NNrunner(object):
	@staticmethod
	def build(self, modelObj, dataObj, cvObj, batchSize = 200, predictSize = 200, iterations = 100, folder = "./", size = (100, 100)):
		modelObj.trainCompile(batchSize)
		modelObj.predictCompile(predictSize)
		for i in xrange(iterations):
			X, index = dataObj.miniBatch(batchSize)
			modelObj.trainCalc(X, X, iteration = 1, debug = True, errorCollect = True)
			X2 = cvObj.X
			E = NNsupport.crossV(batchSize, X2, modelObj)
			print "\n" + str(E) + "\n", i
		NNsupport.errorG(modelObj.errorArray, folder + "g/", 10)
		modelObj.modelSaver(folder + "m/")
		modelObj.weightsVisualizer(folder + "w/", size, color = "L")
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#