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
from TheanoNNclassCORE2 import *		# Some cool NN builder here
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Options store creation
#---------------------------------------------------------------------#
#-----#
OPTIONS2 = OptionsStore(learnStep = 0.01, regularization = False, sparsity = False, lamda = 3e-3, rmsProp = False, decay = 0.9, dropout = False, dropOutParam = (0.75, 1, 0.5, 0.5), mmsmin = 1e-11)
#OPTIONS2 = OptionsStore(learnStep = 0.01, regularization = False, sparsity = False, lamda = 3e-3, rmsProp = False, decay = 0.9, dropout = True, dropOutParam = (0.75, 1, 0.5, 0.5), mmsmin = 1e-11)
#OPTIONS2 = OptionsStore(learnStep = 0.01, regularization = False, sparsity = False, lamda = 3e-3, rmsProp = True, decay = 0.9, dropout = True, dropOutParam = (0.75, 0.5), mmsmin = 1e-11)
OPTIONS2.Printer()
#-----#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Data loader
#---------------------------------------------------------------------#
# SCIKIT
#---------------------------------------------------------------------#
#-----#
DATA = csvDataLoader("./KaggleSCI/train.csv", startColumn = 0, skip = 0)
#DATA.X = DataMutate.deNormalizer(DataMutate.Normalizer(DATA.X), afterzero = 30)
print DATA.X.shape, DATA.Y.shape
#-----#
DATA2 = csvDataLoader("./KaggleSCI/test.csv", startColumn = 0, skip = 0)
#DATA2.X = DataMutate.deNormalizer(DataMutate.Normalizer(DATA2.X), afterzero = 30)
print DATA2.X.shape, DATA2.Y.shape
#-----#
DATA3 = csvDataLoader("./KaggleSCI/trainLabels.csv", startColumn = 1, skip = 0)
print DATA3.X.shape, DATA3.Y.shape
#-----#
aeTRAIN = multiData(DATA, DATA2)
print aeTRAIN.X.shape, aeTRAIN.Y.shape
#-----#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Training softmax
#---------------------------------------------------------------------#
#-----#
batchSize = 1000
#-----#
#NN = TheanoNNclass((DATA.input, 256, 128, 2), OPTIONS2, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.SoftMax))
NN = TheanoNNclass((DATA.input, 256, 128, 2), OPTIONS2, modelFunction = (FunctionModel.MaxOut, FunctionModel.MaxOut, FunctionModel.SoftMax))
#-----#
NN.trainCompile(batchSize)
NN.predictCompile(batchSize)
#-----#
for i in xrange(1500):
	X, index = DATA.miniBatch(batchSize)
	Y = np.tile(DATA3.Y[:, index], (2, 1))
	M = np.tile([0,1], (batchSize, 1)).T
	Y = (Y == M) * 1.0
	print Y.shape, X.shape
	NN.trainCalc(X, Y, iteration = 1, debug = True, errorCollect = True)
	print i
NNsupport.errorG(NN.errorArray, "./GRA/ololo5", 10)
NN.modelSaver("./KaggleMNI/SM.txt")
#-----#
"""
#-----#
AE = TheanoNNclass((aeTRAIN.input, 1024, aeTRAIN.input), OPTIONS2)
AE.modelLoader("./KaggleSCI/AE_F.txt")
AEL = AE.paramGetter()
#-----#
batchSize = 200
#-----#
NN = TheanoNNclass((DATA.input, 1024, 512, 256, 2), OPTIONS2, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.SoftMax))
NNL = NN.paramGetter()
NNL[0:2] = AEL[0:2]
NN.paramSetter(NNL)
NN.trainCompile(batchSize)
NN.predictCompile(batchSize)
for i in xrange(10000):
	X, index = DATA.miniBatch(batchSize)
	Y = np.tile(DATA3.Y[:, index], (2, 1))
	M = np.tile([0,1], (batchSize, 1)).T
	Y = (Y == M) * 1.0
	print Y.shape, X.shape
	NN.trainCalc(X, Y, iteration = 1, debug = True)
	print i
NN.modelSaver("./KaggleSCI/SM2.txt")
#-----#
"""
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Using softmax glue ae
#---------------------------------------------------------------------#
"""
#-----#
#NN = TheanoNNclass((DATA.input, 256, 128, 2), OPTIONS2, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.SoftMax))
NN = TheanoNNclass((DATA.input, 1024, 512, 256, 2), OPTIONS2, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.SoftMax))
#NN.modelLoader("./KaggleSCI/SM.txt")
NN.modelLoader("./KaggleSCI/SM2.txt")
NN.predictCompile(DATA2.number)
#-----#
VECTOR = np.argmax(NN.predictCalc(DATA2.X, debug = True).out, axis = 0).T
file = open("./KaggleSCI/PRED2.txt", "a")
file.write("Id,Solution\n")
for i in xrange(len(VECTOR)):
	file.write(str(i + 1) + "," + str(VECTOR[i]) + "\n")
file.close()
#-----#
"""
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#