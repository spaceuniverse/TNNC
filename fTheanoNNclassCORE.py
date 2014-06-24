#---------------------------------------------------------------------#
#------------------------------------------------CORE-7.1-------------#
#---------------------------------------------------------------------#
# http://en.wikipedia.org/wiki/Harder,_Better,_Faster,_Stronger
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
import time					# What time is it? A.T.
#import datetime
import matplotlib.pyplot as plt
from fCutClassCORE import *				# Convolution cut images class
#---------------------------------------------------------------------#
# Data mutation and calculation functions
#---------------------------------------------------------------------#
class DataMutate(object):
	@staticmethod
	def deNormalizer(ia, afterzero = 20):		# Mapped 0-255 to 0-1 and round to 5 digit after zero
		ia = np.array(ia)
		ia = np.around(ia / 255.0, decimals = afterzero)
		return ia
	@staticmethod
	def Normalizer(ia):				# Mapped to 0-255
		min = np.min(ia)
		max = np.max(ia)
		koeff = 255 / (max - min)
		ia = (ia - min) * koeff
		return ia
	@staticmethod
	def PCAW(X, epsilon = 0.01):			# PCA Whitening. One picture for now
		M = X.mean(axis = 0)
		X = X - M
		C = dot(X, X.T)			# / size(x, 1) for not only one picture
		U, S, V = linalg.svd(C)
		# Original formula: xZCAwhite = U * diag(1.0 / sqrt(diag(S) + epsilon)) * U' * x;
		# http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
		ex1 = diag(1.0 / sqrt(S + epsilon))
		ex2 = dot(U, ex1)
		ex3 = dot(ex2, U.T)
		xPCAWhite = dot(ex3, X)
		# Second way
		# V, D = eigh(C)
		# ex1 = diag(1.0 / sqrt(V + epsilon))
		# ex2 = dot(D, ex1)
		# ex3 = dot(ex2, D.T)
		# xPCAWhite = dot(ex3, X)
		return xPCAWhite
#---------------------------------------------------------------------#
# Support functions
#---------------------------------------------------------------------#
class Graphic(object):
	@staticmethod
	def PicSaver(img, folder, name, color = "L"):			# Saves picture to folder. Color "L" or "RGB"
		imsave = Image.fromarray(DataMutate.Normalizer(img))		# Normalizer(img).astype('uint8') for RGB
		imsave = imsave.convert(color)
		imsave.save(folder + name + ".jpg", "JPEG", quality = 100)
#---------------------------------------------------------------------#
# Data workers
#---------------------------------------------------------------------#
class BatchMixin(object):
	REPORT = "OK"
	def miniBatch(self, number):			# Method for minibatch return
		minIndex = np.random.randint(0, self.number, number)
		self.miniX = self.X[:, minIndex]
		return self.miniX, minIndex
#---------------------------------------------------------------------#
class cPicleDataLoader(BatchMixin):			# Data loader from cPicle file
	def __init__(self, folder):			# folder - Path to cPicle file which contains array with all pictures in grayscale, each row is a picture
		dataload = open(folder, "rb")
		data = cPickle.load(dataload)
		dataload.close()
		data = np.array(data)
		data = data.astype('float')		# Each row is a picture reshaped in vector of features
		self.X = data.T			# Each column is a picture reshaped in vector of features
		self.number = len(data)		# Number of pictures in array
		self.input = len(self.X)		# Number of pixels in one picture, basically this equals to multiplication of dimensions such as 100 * 100 (x * y)
#---------------------------------------------------------------------#
class csvDataLoader(BatchMixin):				# Data loader from csv file
	def __init__(self, folder, startColumn = 1, skip = 1):
		data = np.loadtxt(open(folder, "rb"), delimiter = ",", skiprows = skip)
		data = data.astype('float')
		if len(data.shape) == 1:		# Fixed (1000,) bug
			data = np.reshape(data, (data.shape[0], 1))
		self.X = data[:, startColumn:].T
		self.Y = data[:, 0:startColumn].T
		self.number = len(data)
		self.input = len(self.X)
#---------------------------------------------------------------------#
class multiData(BatchMixin):				# Glues data in one block
	def __init__(self, *objs):
		xtuple = ()
		ytuple = ()
		for obj in objs:
			xtuple += (obj.X,)
			ytuple += (obj.Y,)
		self.X = np.concatenate(xtuple, axis = 1)
		self.Y = np.concatenate(ytuple, axis = 1)
		self.number = self.X.shape[1]
		self.input = self.X.shape[0]
#---------------------------------------------------------------------#
# Activation functions
#---------------------------------------------------------------------#
class FunctionModel(object):
	@staticmethod				# FunctionModel.Sigmoid
	def Sigmoid(W, X, B, E, *args):			# E - ein. Hack for bias. You should remember nobody perfect
		z = T.dot(W, X) + T.dot(B, E)
		a = 1 / (1 + T.exp(-z))
		return a
	@staticmethod				# FunctionModel.Tanh
	def Tanh(W, X, B, E, *args):
		z = T.dot(W, X) + T.dot(B, E)
		a = (T.exp(z) - T.exp(-z)) / (T.exp(z) + T.exp(-z))
		return a
	@staticmethod				# FunctionModel.SoftMax
	def SoftMax(W, X, B, E, *args):
		z = T.dot(W, X) + T.dot(B, E)
		numClasses = W.get_value().shape[0]
		# ___CLASSIC___ #
		# a = T.exp(z) / T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z), axis = 0)])
		# _____________ #
		# Second way antinan
		# a = T.exp(z - T.log(T.sum(T.exp(z))))
		# a = T.exp(z - T.log(T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z), axis = 0)])))		#FIXED?
		# ___ANTINAN___ #
		z_max = T.max(z, axis = 0)
		a = T.exp(z - T.log(T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z - z_max), axis = 0)])) - z_max)
		# _____________ #
		# Some hacks for fixing float32 GPU problem
		# a = T.clip(a, float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max))
		# a = T.clip(a, 1e-20, 1e20)
		# http://www.velocityreviews.com/forums/t714189-max-min-smallest-float-value-on-python-2-5-a.html
		# http://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html
		# Links about possible approaches to fix nan
		# http://blog.csdn.net/xceman1997/article/details/9974569
		# https://github.com/Theano/Theano/issues/1563
		return a
	@staticmethod				# FunctionModel.MaxOut
	def MaxOut(W, X, B, E, *args):
		z = T.dot(W, X) + T.dot(B, E)
		d = T.shape(z)
		n_elem = args[0]
		z = z.reshape((d[0] / n_elem, n_elem, d[1]))
		a = T.max(z, axis = 1)
		return a
	"""
	@staticmethod				# FunctionModel.MaxOut OLD IMPLEMENTATION
	def MaxOut(W, X, B, E, *args):
		numPatches = E.get_value().shape[1]
		A = np.zeros((1, numPatches)).astype(theano.config.floatX)
		neurons = T.arange(T.shape(W)[0])
		# Without B
		# components, updates = theano.scan(fn = lambda n, weights, data, activation: T.max(data.T * T.dot(T.shape_padleft(weights[n, :]).T, E).T, axis = 1), sequences = neurons, non_sequences = [W, X, A], outputs_info = None)
		# From theano manual: T.concatenate([x0, x1[0], T.shape_padright(x2)], axis = 1)
		# T.dot(bias[n, :], E).T
		# ___ConcatenateB___ #
		# components, updates = theano.scan(fn = lambda n, weights, bias, data, activation: T.max(T.concatenate([data.T * T.dot(T.shape_padleft(weights[n, :]).T, E).T, T.dot(T.shape_padleft(bias[n, :]), E).T], axis = 1), axis = 1), sequences = neurons, non_sequences = [W, B, X, A], outputs_info = None)
		components, updates = theano.scan(fn = lambda n, weights, bias, data, activation: T.max(T.concatenate([data.T * T.dot(T.shape_padleft(weights[n, :]).T, E).T, (bias[n, 0] * E).T], axis = 1), axis = 1), sequences = neurons, non_sequences = [W, B, X, A], outputs_info = None)
		# __________________ #
		# Piece of code: maxot = theano.function(inputs = [X, A], outputs = components)
		# res = maxot(X, np.zeros((1, X.shape[0]), dtype = np.float64))
		return components
	"""
#---------------------------------------------------------------------#
# Options instance
#---------------------------------------------------------------------#
class OptionsStore(object):
	def __init__(self, learnStep = 0.01, regularization = False, lamda = 0.001, sparsity = False, sparsityParam = 0.01, beta = 0.01, dropout = False, dropOutParam = (1, 1), rmsProp = False, decay = 0.9, mmsmin = 1e-10, rProp = False, dropconnect = False, dropConnectParam = (1, 1), pool_size = None):
		self.learnStep = learnStep		# Learning step for gradient descent
		self.regularization = regularization	# Weight decay on|off
		self.lamda = lamda			# Weight decay coef
		self.sparsity = sparsity		# Sparsity on|off
		self.sparsityParam = sparsityParam
		self.beta = beta
		self.dropout = dropout		# Dropout on|off
		self.dropOutParam = dropOutParam		# dropOutParam = (1, 0.7, 0,5) etc.
		self.rmsProp = rmsProp		# rmsProp on|off
		self.decay = decay
		self.mmsmin = mmsmin			# Min mms value
		self.rProp = rProp			# For full batch only
		self.dropconnect = dropconnect		# DropConnect http://cs.nyu.edu/~wanli/dropc/dropc.pdf or http://cs.nyu.edu/~wanli/dropc/
		self.dropConnectParam = dropConnectParam	# dropConnectParam = (1, 1, 0.5) etc.
		self.pool_size = pool_size		# For MaxOut pooling region
	def Printer(self):
		print self.__dict__
#---------------------------------------------------------------------#
# Basic neuralnet class
#---------------------------------------------------------------------#
class TheanoNNclass(object):				# Basic NN builder
	# TheanoNNclass(architecture, options, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid))
	# Main class for building neural net with different configurations
	# Parameters
	# ----------
	# architecture : tuple
	#	Number of neurons in layers. Example: (100, 20, 10)
	# options : OptionsStore
	#	The options
	# modelFunction : function
	#	Optional	
	REPORT = "OK"
	def __init__(self, architecture, options, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid)):
		self.architecture = architecture		# Architecture - number of neurons in layers, tuple. Example: (100, 20, 10)
		self.options = options		# Obj with options of nn train
		self.modelFunction = modelFunction
		self.lastArrayNum = len(architecture) - 1
		random = sqrt(6) / sqrt(architecture[0] + architecture[self.lastArrayNum])
		self.varArrayW = []
		self.varArrayB = []
		for i in xrange(self.lastArrayNum):
			if self.modelFunction[i] != FunctionModel.MaxOut:
				w = theano.shared((np.random.randn(architecture[i + 1], architecture[i]) * 2 * random - random).astype(theano.config.floatX), name = "w%s" % (i + 1))
			else:
				w = theano.shared((np.random.randn(architecture[i + 1] * self.options.pool_size, architecture[i]) * 2 * random - random).astype(theano.config.floatX), name = "w%s" % (i + 1))
			self.varArrayW.append(w)
			if self.modelFunction[i] != FunctionModel.MaxOut:
				b = theano.shared(np.tile(0.0, (architecture[i + 1], 1)).astype(theano.config.floatX), name = "b%s" % (i + 1))
			else:
				b = theano.shared(np.tile(0.0, (architecture[i + 1] * self.options.pool_size, 1)).astype(theano.config.floatX), name = "b%s" % (i + 1))
			self.varArrayB.append(b)
			self.gradArray = []			# Array for T.grad theano function input, contains array of [w1, b1, w2, b2] etc.
		for i in xrange(self.lastArrayNum):		# Possible use len(self.varArrayB) or len(self.varArrayW) instead
			self.gradArray.append(self.varArrayW[i])
			self.gradArray.append(self.varArrayB[i])			
	def trainCompile(self, numpatches):			# Numpatches - Number of patches in array (pictures, vectors, ect.)
		# Compiles code in c for using on GPU
		# Parameters
		# ----------
		# numpatches : int
		#	Number of samples in array	
		self.ein = theano.shared(np.tile(1.0, (1, numpatches)).astype(theano.config.floatX), name = "ein")		# Dirty hack for bias dimension fix also dropout instead tile
		self.x = T.matrix("x")
		self.y = T.matrix("y")
		if self.options.dropout:
			srng = RandomStreams()				# Theano random generator for dropout
			# self.dropOutParam = self.options.dropOutParam
			self.dropOutVectors = []
			for i in xrange(self.lastArrayNum):
				self.dropOutVectors.append(srng.binomial(p = self.options.dropOutParam[i], size = (self.architecture[i], 1)).astype(theano.config.floatX))
		if self.options.dropconnect:
			srng = RandomStreams()				# Theano random generator for dropconnect
			self.dropConnectVectors = []
			for i in xrange(self.lastArrayNum):
				# STILL NEED TO WRITE
				self.dropConnectVectors.append(srng.binomial(p = self.options.dropConnectParam[i], size = (self.architecture[i + 1], self.architecture[i])).astype(theano.config.floatX))	# Weights
		self.varArrayA = []
		for i in xrange(self.lastArrayNum):
			variable2 = T.dot(self.dropOutVectors[i], self.ein) if self.options.dropout else 1.0
			variable = self.x if i == 0 else self.varArrayA[i - 1]
			a = self.modelFunction[i](self.varArrayW[i], variable * variable2, self.varArrayB[i], self.ein)
			self.varArrayA.append(a)
		self.sparse = 0
		self.regularize = 0						# Weight decay
		if self.options.sparsity and self.lastArrayNum <= 2:			# For now only for nn with one hidden layer
			sprs = T.sum(self.varArrayA[0], axis = 1) / (numpatches + 0.0)
			epsilon = 1e-20
			sprs = T.clip(sprs, epsilon, 1 - epsilon)
			KL = T.sum(self.options.sparsityParam * T.log(self.options.sparsityParam / sprs) + (1 - self.options.sparsityParam) * T.log((1 - self.options.sparsityParam) / (1 - sprs)))
			self.sparse = self.options.beta * KL
		if self.options.regularization:
			wsum = 0
			for w in self.varArrayW:
				wsum += T.sum(w ** 2)
			self.regularize = self.options.lamda / 2 * wsum
		XENT = 1.0 / numpatches * T.sum((self.y - self.varArrayA[-1]) ** 2 * 0.5)
		# Fix float32 GPU and nan in softmax
		# if self.modelFunction[-1] == FunctionModel.SoftMax:
		#	XENT = - 1.0 / numpatches * T.sum(self.y * T.log(self.varArrayA[-1]))
		self.errorArray = []						# Storage for costs
		self.cost = XENT + self.sparse + self.regularize
		self.derivativesArray = []
		self.derivativesArray = T.grad(self.cost, self.gradArray)		# Get derivatives using theano function
		if self.options.rmsProp:
			self.MMSprev = []
			self.MMSnew = []
			for i in xrange(len(self.derivativesArray)):
				mmsp = theano.shared(np.tile(0.0, self.gradArray[i].get_value().shape).astype(theano.config.floatX), name = "mmsp%s" % (i + 1))	# 0.0 - 1.0 maybe
				self.MMSprev.append(mmsp)
				mmsn = self.options.decay * mmsp + (1 - self.options.decay) * self.derivativesArray[i] ** 2
				mmsn = T.clip(mmsn, self.options.mmsmin, 1e+20)	# Fix nan if rmsProp
				self.MMSnew.append(mmsn)
		if self.options.rProp:
			baseRpropStep = 0.001
			decreaseCoef = 0.1
			increaseCoef = 1.3
			minRpropStep = 1e-6
			maxRpropStep = 50
			signedChanged = 0
			numWeights = 0
			for i in xrange(self.lastArrayNum):
				numWeights += self.architecture[i] * self.architecture[i + 1]
			self.prevGW = []
			self.deltaW = []
			for i in xrange(len(self.derivativesArray)):
				prevGW = theano.shared(np.tile(1.0, self.gradArray[i].get_value().shape).astype(theano.config.floatX), name = "prewGW%s" % (i + 1))
				deltaW = theano.shared(np.tile(np.float32(baseRpropStep), self.gradArray[i].get_value().shape).astype(theano.config.floatX), name = "deltaW%s" % (i + 1))
				self.prevGW.append(prevGW)
				self.deltaW.append(deltaW)
		self.updatesArray = []					# Array for train theano function updates input parameter
		for i in xrange(len(self.derivativesArray)):
			if self.options.rmsProp:
				updateVar = self.options.learnStep * self.derivativesArray[i] / self.MMSnew[i] ** 0.5
				self.updatesArray.append((self.MMSprev[i], self.MMSnew[i]))
			elif self.options.rProp:
				updateVar = self.deltaW[i] * decreaseCoef * T.lt(self.prevGW[i] * self.derivativesArray[i], 0) + self.deltaW[i] * increaseCoef * T.gt(self.prevGW[i] * self.derivativesArray[i], 0)
				updateVar = T.clip(updateVar, minRpropStep, maxRpropStep)
				updateVar = updateVar * T.sgn(self.derivativesArray[i])
				self.updatesArray.append((self.prevGW[i], self.derivativesArray[i]))
				self.updatesArray.append((self.deltaW[i], T.abs_(updateVar)))
				signedChanged += T.sum(T.lt(self.prevGW[i] * self.derivativesArray[i], 0))
			else:
				updateVar = self.options.learnStep * self.derivativesArray[i]
			self.updatesArray.append((self.gradArray[i], self.gradArray[i] - updateVar))
		self.train = theano.function(inputs = [self.x, self.y], outputs = [self.cost], updates = self.updatesArray, allow_input_downcast = True)
		return self
	def predictCompile(self, numpatches, layerNum = -1):				# layerNum - number of output layer. Last one = -1. Example: in 3 layers nn output "1", hidden "0"
		# Numpatches - Number of patches in array (pictures, vectors, ect.)
		self.ein2 = theano.shared(np.tile(1.0, (1, numpatches)).astype(theano.config.floatX), name = "ein2")
		self.data = T.matrix("data")
		self.varArrayAc = []
		for i in xrange(self.lastArrayNum):
			# Old dependent from train approach
			# variable2 = self.dropOutParam[i] if hasattr(self, 'dropOutParam') else 1
			variable2 = self.options.dropOutParam[i] if self.options.dropout else 1.0
			variable = self.data if i == 0 else self.varArrayAc[i - 1]
			a = self.modelFunction[i](self.varArrayW[i] * variable2, variable, self.varArrayB[i], self.ein2)		# Fix ein and num. Just create other ein2 | FIXED, but why | variable * variable2
			self.varArrayAc.append(a)
		self.predict = theano.function(inputs = [self.data], outputs = self.varArrayAc[layerNum], allow_input_downcast = True)	# self.lastArrayNum - 1
		return self
	def trainCalc(self, X, Y, iteration = 10, debug = False, errorCollect = False):	# Need to call trainCompile before
		"""
		if debug:
			for i in xrange(iteration):
				timeStart = datetime.datetime.now()
				error = self.train(X, Y)
				if errorCollect:
					self.errorArray.append(error)
				timeStop = datetime.datetime.now()
				remainingTime = ((timeStop - timeStart) * (iteration - i * 10))
				print "Error: {:014.12f}; ".format(float(error[0])),
				print "Remaining Time: {:}".format(remainingTime)
		else:
			for i in xrange(iteration):
				error = self.train(X, Y)
				if errorCollect:
					self.errorArray.append(error)
		"""
		for i in xrange(iteration):
			timeStart = time.time()
			error = self.train(X, Y)
			if errorCollect:
				self.errorArray.append(error)
			timeStop = time.time()
			if debug:
				print error,
				print round(((timeStop - timeStart) / 60) * (iteration - i), 5)		# In minutes
		return self
	def predictCalc(self, X, debug = False):		# Need to call predictCompile before
		self.out = self.predict(X)		# Matrix of outputs. Each column is a picture reshaped in vector of features
		if debug: print self.out.shape
		return self
	def getStatus(self):				# Its time for troubles
		print self.REPORT
		return self
	def paramGetter(self):			# Returns the values of model parameters such as [w1, b1, w2, b2] ect.
		self.model = []
		counter = 0
		for obj in self.gradArray:
			# FIX FROM PAPER
			L = len(self.gradArray)
			M = range(0, L, 2)
			# OLD variable = self.options.dropOutParam[self.gradArray.index(obj) / 2] if self.gradArray.index(obj) % 2 == 0 and self.options.dropout else 1.0
			variable = self.options.dropOutParam[counter] if counter in M and self.options.dropout else 1.0
			self.model.append(obj.get_value() * variable)
			counter += 1
		return self.model
	def paramSetter(self, array):			# Setups loaded model parameters
		counter = 0
		for obj in self.gradArray:
			# FIX FROM PAPER
			L = len(self.gradArray)
			M = range(0, L, 2)
			# OLD variable = self.options.dropOutParam[self.gradArray.index(obj) / 2] if self.gradArray.index(obj) % 2 == 0 and self.options.dropout else 1.0
			variable = self.options.dropOutParam[counter] if counter in M and self.options.dropout else 1.0
			obj.set_value((array[self.gradArray.index(obj)] / variable).astype(theano.config.floatX))		# np.float32(array) or astype
			counter += 1
	def modelSaver(self, folder):			# In cPickle format in txt file
		f = file(folder, "wb")
		for obj in self.paramGetter():
			cPickle.dump(obj, f, protocol = cPickle.HIGHEST_PROTOCOL)
		f.close()
		self.getStatus()
		return self
	def modelLoader(self, folder):			# Path to model txt file
		f = file(folder, "rb")
		loadedObjects = []
		for i in xrange(len(self.gradArray)):	# Array that contains w1 b1 w2 b2 etc.
			loadedObjects.append(cPickle.load(f))
		f.close()				# Then we need to update W and B parameters
		self.paramSetter(loadedObjects)
		self.getStatus()
		return self
	def weightsVisualizer(self, folder, size = (100, 100), color = "L"):	# For now only for first layer. Second in test mode
		W1 = self.gradArray[0].get_value()
		W2 = self.gradArray[2].get_value()			# Second layer test. Weighted linear combination of the first layer bases
		for w in xrange(len(W1)):
			img = W1[w, :].reshape(size[0], size[1])		# Fix to auto get size TODO
			Graphic.PicSaver(img, folder, "L1_" + str(w), color)
		for w in xrange(len(W2)):
			img = np.dot(W1.T, W2[w, :]).reshape(size[0], size[1])
			Graphic.PicSaver(img, folder, "L2_" + str(w), color)
		return self
#---------------------------------------------------------------------#
# Usefull functions
#---------------------------------------------------------------------#
class NNsupport(object):
	@staticmethod
	def crossV(number, data, modelObj):
		# FIX FOR NOT ONLY AE
		ERROR = 1.0 / number * np.sum((data - modelObj.predictCalc(data).out) ** 2 * 0.5)
		return ERROR
	@staticmethod
	def errorG(errorArray, folder, plotsize = 50):
		x = range(len(errorArray))
		y = list(errorArray)
		area = plotsize
		# Old: plt.scatter(x, y, s = area, alpha = 0.5)
		# plt.show()
		fig = plt.figure(figsize = (10, 8))
		ax = fig.add_subplot(1, 1, 1)		# One row, one column, first plot
		ax.scatter(x, y, s = area, alpha = 0.5)
		fig.savefig(folder)
#---------------------------------------------------------------------#
# Can this really be the end? Back to work you go again
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#