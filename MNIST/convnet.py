import numpy as np
import gzip



## Forward and Bacward Propogation
def ConvNet(image, label, filt1, filt2, bias1, bias2, theta, thetabias):
	

	####################################   Forward Propogation   #############################################################

	## First Convolution layer
		
	(d, w, w) = image.shape		
	d1 = len(filt1)
	d2 = len(filt2)
	( _, f, f) = filt1[0].shape
	w1 = w-f+1
	w2 = w1-f+1
	
	
	layer1 = np.zeros((d1,w1,w1))
	layer2 = np.zeros((d2,w2,w2))

	for i in range(0,d1):
		for j in range(0,w1):
			for k in range(0,w1):
				layer1[i,j,k] = np.sum(image[:,j:j+f,k:k+f]*filt1[i])+bias1[i]
	#relu activation	
	layer1[layer1<=0] = 0 


	## Second Convolution layer
	for i in range(0,d2):
		for j in range(0,w2):
			for k in range(0,w2):
				layer2[i,j,k] = np.sum(layer1[:,j:j+f,k:k+f]*filt2[i])+bias2[i]
	# relu activation
	layer2[layer2<=0] = 0 

	## Pooling layer [filter size 2*2 and stride size 2,2]
	pooling_layer = maxpool(layer2, 2, 2)	

	fc = pooling_layer.reshape(((w2//2)*(w2//2)*d2,1))
	
	out = theta.dot(fc) + thetabias	
	
	# Apply Softmax
	cost, prob = softmax(out, label)
	if np.argmax(out)==np.argmax(label):
		acc=1
	else:
		acc=0

	################################################   Backward Propogation    ################################################

	gradientout = prob - label	
	
	gradienttheta = gradientout.dot(fc.T)

	gradientthetabias = sum(gradientout.T).T.reshape((10,1))			

	gradientfc = theta.T.dot(gradientout)		

	gradientpool = gradientfc.T.reshape((d2, w2//2, w2//2))

	gradientlayer2 = np.zeros((d2, w2, w2))
	
	#Getting the Indices of Conv Layer for Unpooling and Calculating the Gradient Values for layer2
	for i in range(0,d2):
		j=0
		while(j<w2):
			k=0
			while(k<w2):
				(a,b) = argmax(layer2[i,j:j+2,k:k+2]) ## Getting indexes of maximum value in the array
				gradientlayer2[i,j+a,k+b] = gradientpool[i,j//2,k//2]
				k+=2
			j+=2
	
	gradientlayer2[layer2<=0]=0
	
	#Initializing The gradient Variables with Zeros
	gradientlayer1 = np.zeros((d1, w1, w1))
	gradientfilt2 = {}
	gradientbias2 = {}
	for s in range(0,d2):
		gradientfilt2[s] = np.zeros((d1,f,f))
		gradientbias2[s] = 0

	gradientfilt1 = {}
	gradientbias1 = {}
	for t in range(0,d1):
		gradientfilt1[t] = np.zeros((d,f,f))
		gradientbias1[t] = 0

	#Calculating the Gradient Values for filt2, bias2, layer1
	for i in range(0,d2):
		for j in range(0,w2):
			for k in range(0,w2):
				gradientfilt2[i]+=gradientlayer2[i,j,k]*layer1[:,j:j+f,k:k+f]
				gradientlayer1[:,j:j+f,k:k+f]+=gradientlayer2[i,j,k]*filt2[i]
		gradientbias2[i] = np.sum(gradientlayer2[i])
	gradientlayer1[layer1<=0]=0

	#Calculating the Gradient Values for filt1, bias1
	for i in range(0,d1):
		for j in range(0,w1):
			for k in range(0,w1):
				gradientfilt1[i]+=gradientlayer1[i,j,k]*image[:,j:j+f,k:k+f]

		gradientbias1[i] = np.sum(gradientlayer1[i])

	#Returning The Values
	return [gradientfilt1, gradientfilt2, gradientbias1, gradientbias2, gradienttheta, gradientthetabias, cost, acc]



## Batch wise Training of Images
def MGD(batch, learningRate, w, l, mu, filt1, filt2, bias1, bias2, theta, thetabias, cost, acc):
	
	#Considering a Bath of Training Data
	X = batch[:,0:-1]
	X = X.reshape(len(batch), l, w, w)
	y = batch[:,-1]

	# Defining Temporary variables
	n_acc=0
	cost_ = 0
	batchSize = len(batch)
	gradientfilt2 = {}
	gradientfilt1 = {}
	gradientbias2 = {}
	gradientbias1 = {}
	temp1 = {}
	temp2 = {}
	btemp1 = {}
	btemp2 = {}

	# Initializing the dfilter values with zeros
	for k in range(0,len(filt2)):
		gradientfilt2[k] = np.zeros(filt2[0].shape)
		gradientbias2[k] = 0
		temp2[k] = np.zeros(filt2[0].shape)
		btemp2[k] = 0
	for k in range(0,len(filt1)):
		gradientfilt1[k] = np.zeros(filt1[0].shape)
		gradientbias1[k] = 0
		temp1[k] = np.zeros(filt1[0].shape)
		btemp1[k] = 0
	gradienttheta = np.zeros(theta.shape)
	gradientthetabias = np.zeros(thetabias.shape)
	temp3 = np.zeros(theta.shape)
	btemp3 = np.zeros(thetabias.shape)


	# Batch Training

	for i in range(0,batchSize):
		
		img = X[i]

		labels = np.zeros((theta.shape[0],1))
		labels[int(y[i]),0] = 1
		
		## Calculating gradient for the parameters by ConvNet Function
		[gradientfilt1_, gradientfilt2_, gradientbias1_, gradientbias2_, gradienttheta_, gradientthetabias_, New_cost, acc_] = ConvNet(img, labels, filt1, filt2, bias1, bias2, theta, thetabias)


		# Transfering the gradients to the Initialized variables
		for j in range(0,len(filt2)):
			gradientfilt2[j]+=gradientfilt2_[j]
			gradientbias2[j]+=gradientbias2_[j]
		for j in range(0,len(filt1)):
			gradientfilt1[j]+=gradientfilt1_[j]
			gradientbias1[j]+=gradientbias1_[j]
		gradienttheta+=gradienttheta_
		gradientthetabias+=gradientthetabias_

		cost_+=New_cost
		n_acc+=acc_
		
	#Updating the Weights

	for j in range(0,len(filt1)):
		temp1[j] = mu*temp1[j] -learningRate*gradientfilt1[j]/batchSize
		filt1[j] += temp1[j]
		btemp1[j] = mu*btemp1[j] -learningRate*gradientbias1[j]/batchSize
		bias1[j] += btemp1[j]

	for j in range(0,len(filt2)):
		temp2[j] = mu*temp2[j] -learningRate*gradientfilt2[j]/batchSize
		filt2[j] += temp2[j]
		btemp2[j] = mu*btemp2[j] -learningRate*gradientbias2[j]/batchSize
		bias2[j] += btemp2[j]

	temp3 = mu*temp3 - learningRate*gradienttheta/batchSize
	theta += temp3

	btemp3 = mu*btemp3 -learningRate*gradientthetabias/batchSize
	thetabias += btemp3
	
	#Updating the Cost and Accuracy

	cost_ = cost_/batchSize
	cost.append(cost_)
	accuracy = float(n_acc)/batchSize
	acc.append(accuracy)

	return [filt1, filt2, bias1, bias2, theta, thetabias, cost, acc]


#Extracting_Images
def extract_Images(files, NoofImages, ImageWidth):

	print('Extract_Images', files)
	with gzip.open(files) as bytestream:
		bytestream.read(16)
		bufer = bytestream.read(ImageWidth * ImageWidth * NoofImages)
		data = np.frombuffer(bufer, dtype=np.uint8).astype(np.float32)
		data = data.reshape(NoofImages,ImageWidth*ImageWidth)
		return data

#Extracting_labels
def extract_labels(files, NoofImages):
	
	print('Extract_labels', files)
	with gzip.open(files) as bytestream:
		bytestream.read(8)
		bufer = bytestream.read(1 * NoofImages)
		labels = np.frombuffer(bufer, dtype=np.uint8).astype(np.int64)
	return labels

# Maximum value of the array
def argmax(h):
	index = np.argmax(h, axis=None)
	multi_index = np.unravel_index(index, h.shape)
	if np.isnan(h[multi_index]):
		nancount = np.sum(np.isnan(h))
		idx = np.argpartition(h, -nancount-1, axis=None)[-nancount-1]
		multi_index = np.unravel_index(index, h.shape)
	return multi_index

# Predict class 
def predict(image, filt1, filt2, bias1, bias2, theta, thetabias):
	


	(d,w,w)=image.shape
	(d1,f,f) = filt2[0].shape
	d2 = len(filt2)
	w1 = w-f+1
	w2 = w1-f+1
	layer1 = np.zeros((d1,w1,w1))
	layer2 = np.zeros((d2,w2,w2))

	#First Convolution Layer
	for i in range(0,d1):
		for j in range(0,w1):
			for k in range(0,w1):
				layer1[i,j,k] = np.sum(image[:,j:j+f,k:k+f]*filt1[i])+bias1[i]
	layer1[layer1<=0] = 0 

	#First Convolution Layer
	for i in range(0,d2):
		for j in range(0,w2):
			for k in range(0,w2):
				layer2[i,j,k] = np.sum(layer1[:,j:j+f,k:k+f]*filt2[i])+bias2[i]
	layer2[layer2<=0] = 0

	#Pooling
	pooling_layer = maxpool(layer2, 2, 2)

	#Fully Connected Layer	
	fc = pooling_layer.reshape(((w2//2)*(w2//2)*d2,1))
	out = theta.dot(fc) + thetabias	
	expout = np.exp(out, dtype=np.float)
	prob = expout/sum(expout)

	return np.argmax(prob), np.max(prob)


#Maxpool Function
def maxpool(X, f, st):
	(d, w, w) = X.shape
	pool = np.zeros((d, (w-f)//st+1,(w-f)//st+1))
	for i in range(0,d):
		j=0
		while(j<w):
			k=0
			while(k<w):
				pool[i,j//2,k//2] = np.max(X[i,j:j+f,k:k+f])
				k+=st
			j+=st
	return pool

#Softmax Classifier Function
def softmax(out,y):
	expout = np.exp(out, dtype=np.float)
	prob = expout/sum(expout)
	
	p = sum(y*prob)
	cost = -np.log(p)	
	return cost,prob	


