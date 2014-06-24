#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Some examples of useful code. No, really, i'm not joking
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numpy import *
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T
import cPickle
import time			# What time is it? Adven...
from fTheanoNNclassCORE import *	# Some cool NN builder here
#from fWrapperClassCORE import *		# Some cool NN wrapper here
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Options store creation
#---------------------------------------------------------------------#
#-----#
OPTIONS = OptionsStore(learnStep = 0.001, regularization = True, sparsity = True, lamda = 3e-7, sparsityParam = 0.05, beta = 7, rmsProp = True, decay = 0.9, dropout = True, dropOutParam = (0.75, 0.5), mmsmin = 1e-20, rProp = True)
OPTIONS.Printer()
#-----#
#OPTIONS2 = OptionsStore(learnStep = 0.001, regularization = False, sparsity = False, lamda = 3e-3, rmsProp = True, decay = 0.9, dropout = True, dropOutParam = (0.75, 1.0, 0.3, 0.3), mmsmin = 1e-20)
#OPTIONS2.Printer()
#-----#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Data loader
#---------------------------------------------------------------------#
# MNIST
#---------------------------------------------------------------------#
#-----#
DATA3 = csvDataLoader("./KaggleMNI/test.csv", startColumn = 0)
DATA3.X = DataMutate.deNormalizer(DATA3.X, afterzero = 30)
print DATA3.X.shape, DATA3.Y.shape
#-----#
"""
DATA4 = csvDataLoader("./KaggleMNI/train.csv", startColumn = 1)
DATA5 = csvDataLoader("./KaggleMNI/train.csv", startColumn = 1)
DATA4.X = DataMutate.deNormalizer(DATA4.X, afterzero = 30)[:,:30000]
DATA4.Y = DataMutate.deNormalizer(DATA4.X, afterzero = 30)[:,:30000]
DATA5.X = DataMutate.deNormalizer(DATA5.X, afterzero = 30)[:,30000:]
DATA5.Y = DataMutate.deNormalizer(DATA5.X, afterzero = 30)[:,30000:]
print DATA4.X.shape, DATA4.Y.shape
print DATA5.X.shape, DATA5.Y.shape
"""
#-----#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Training autoencoder
#---------------------------------------------------------------------#
# miniBatch
#---------------------------------------------------------------------#

#-----#
AE = TheanoNNclass((DATA4.input, 512, DATA4.input), OPTIONS)
batchSize = 200
AE.trainCompile(batchSize)
AE.predictCompile(batchSize)
#-----#
for i in xrange(90000):
	X, index = DATA4.miniBatch(batchSize)
	AE.trainCalc(X, X, iteration = 1, debug = True)
	X2, index = DATA3.miniBatch(batchSize)
	E = NNsupport.crossV(batchSize, X2, AE)
	NNsupport.errorG(AE.errorArray, "./GRA/" + str(i), 10)
	print "\n" + str(E) + "\n"
	print i
AE.modelSaver("./KaggleMNI/AE_NEW_VERSION.txt")
AE.weightsVisualizer("./W/", size = (28, 28), color = "L")
#-----#

#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Training softmax glue ae
#---------------------------------------------------------------------#
"""
#-----#
AE = TheanoNNclass((DATA4.input, 512, DATA4.input), OPTIONS)
AE.modelLoader("./KaggleMNI/AE_NEW_VERSION.txt")
AEL = AE.paramGetter()
#-----#
batchSize = 200
#-----#
NN = TheanoNNclass((DATA4.input, 512, 362, 362, 10), OPTIONS2, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.SoftMax))
NNL = NN.paramGetter()
NNL[0:2] = AEL[0:2]
NN.paramSetter(NNL)
#-----#
NN.trainCompile(batchSize)
NN.predictCompile(batchSize)
#-----#
for i in xrange(90000):
	X, index = DATA4.miniBatch(batchSize)
	Y = np.tile(DATA4.Y[:, index], (10, 1))
	M = np.tile([0,1,2,3,4,5,6,7,8,9], (batchSize, 1)).T
	Y = (Y == M) * 1.0
	print Y.shape, X.shape
	NN.trainCalc(X, Y, iteration = 1, debug = True)
	print i
NN.modelSaver("./KaggleMNI/SM_NEW_VERSION.txt")
#-----#
"""
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Using softmax glue ae
#---------------------------------------------------------------------#
"""
#-----#
NN = TheanoNNclass((DATA3.input, 512, 362, 362, 10), OPTIONS2, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.SoftMax))
NN.modelLoader("./KaggleMNI/SM_NEW_VERSION.txt")
NN.trainCompile(DATA3.number)
NN.predictCompile(DATA3.number)
#-----#
VECTOR = np.argmax(NN.predictCalc(DATA3.X, debug = True).out, axis = 0).T
file = open("./KaggleMNI/PRED_NEW_VERSION.txt", "a")
file.write("ImageId,Label\n")
for i in xrange(len(VECTOR)):
	file.write(str(i + 1) + "," + str(VECTOR[i]) + "\n")
file.close()
#-----#
"""
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#-----#
"""
#-----#
OPTIONS3 = OptionsStore(learnStep = 0.05, regularization = False, sparsity = False, lamda = 3e-3, rmsProp = True, decay = 0.9, dropout = False, dropOutParam = (0.8, 0.5, 0.5), mmsmin = 1e-9)
OPTIONS3.Printer()
#-----#
batchSize = 100
#-----#
NN = TheanoNNclass((DATA4.input, 512, 128, 10), OPTIONS3, modelFunction = (FunctionModel.MaxOut, FunctionModel.MaxOut, FunctionModel.SoftMax))
#-----#
NN.trainCompile(batchSize)
NN.predictCompile(batchSize)
#-----#
for i in xrange(500):
	X, index = DATA4.miniBatch(batchSize)
	Y = np.tile(DATA4.Y[:, index], (10, 1))
	M = np.tile([0,1,2,3,4,5,6,7,8,9], (batchSize, 1)).T
	Y = (Y == M) * 1.0
	print Y.shape, X.shape
	NN.trainCalc(X, Y, iteration = 1, debug = True, errorCollect = True)
	print i
NNsupport.errorG(NN.errorArray, "./GRA/mnist2", 10)
#-----#
"""
#-----#