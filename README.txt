#---------------------------------------------------------------------#
# For MNIST on Kaggle
#---------------------------------------------------------------------#
# AE
#---------------------------------------------------------------------#
Model:
{
	NN: (28*28, 512, 28*28)
}
Train:
{
	iteration: 900000
	batchSize: 200
	learnStep: 0.001
	regularization: True
	sparsity: True
	lamda: 3e-7
	sparsityParam: 0.05
	beta: 7
	rmsProp: True
	decay: 0.9
	dropout: True
	dropOutParam: (0.75, 0.5)
}
#---------------------------------------------------------------------#
# SM
#---------------------------------------------------------------------#
Model:
{
	NN: (28*28, 512, 362, 362, 10)
	modelFunction: (FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.Sigmoid, FunctionModel.SoftMax)	
}
Train:
{
	iteration: 2*900000
	batchSize: 200
	learnStep: 0.001
	regularization: False
	sparsity: False
	rmsProp: True
	decay: 0.9
	dropout: True
	dropOutParam: (0.75, 1.0, 0.3, 0.3)
}
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#