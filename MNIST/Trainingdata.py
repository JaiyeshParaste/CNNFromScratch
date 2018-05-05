import numpy as np
import pickle
import time
import random
from convnet import *
import scipy.io as sio
from matplotlib import gridspec
import matplotlib.pyplot as plt



#Hyperparameters Setting

NOutput= 10
Mu = 0.95
learning_Rate = 0.01	
ImageWidth= 28
ImageDepth = 1
FilterSize=5
NoFilter1 = 8
NoFilter2 = 8
batchSize = 20
epochs = 2	 






# Extraction of MNIST Training Data 

NooftrainingImages =50000
X = extract_Images('train-images-idx3-ubyte.gz', NooftrainingImages, ImageWidth)
y = extract_labels('train-labels-idx1-ubyte.gz', NooftrainingImages).reshape( NooftrainingImages,1)
X-= int(np.mean(X))
X/= int(np.std(X))
train_data = np.hstack((X,y))
train_data = train_data[:50,:]

# Extraction of MNIST Training Data 

NooftestingImages =10000
X = extract_Images('t10k-images-idx3-ubyte.gz', NooftestingImages, ImageWidth)
y = extract_labels('t10k-labels-idx1-ubyte.gz', NooftestingImages).reshape(NooftestingImages,1)
X-= int(np.mean(X))
X/= int(np.std(X))
test_data = np.hstack((X,y))
test_data = test_data[:1,:]


pickle_file = 'output.pickle'



filt1 = {}
bias1 = {}
filt2 = {}
bias2 = {}

cost = []
accuracy = []



for r in range(0,NoFilter1):

	bias1[r] = 0
	temp = FilterSize*FilterSize*ImageDepth
	filt1[r] = np.random.normal(loc = 0, scale = (1.0) * np.sqrt(1./temp), size = (ImageDepth,FilterSize,FilterSize) )
	


for r in range(0,NoFilter2):

	bias2[r] = 0
	temp = FilterSize*FilterSize*NoFilter2
	filt2[r] = np.random.normal(loc = 0, scale = (1.0) * np.sqrt(1./temp), size = (NoFilter2,FilterSize,FilterSize) )
	



np.random.shuffle(train_data)
NoofImages = train_data.shape[0]

w1 = ImageWidth-FilterSize+1
w2 = w1-FilterSize+1

theta = 0.01*np.random.rand(NOutput, (w2//2)*(w2//2)*NoFilter2)
thetaBias = np.zeros((NOutput,1))







#CostTrack = []
for epoch in range(0,epochs):
	np.random.shuffle(train_data)
	batches = [train_data[k:k + batchSize] for k in range(0, NoofImages, batchSize)]
	t=0
	
	for batch in batches:
		
		
		out = MGD(batch, learning_Rate, ImageWidth, ImageDepth, Mu, filt1, filt2, bias1, bias2, theta, thetaBias, cost, accuracy)
		[filt1, filt2, bias1, bias2, theta, thetaBias, cost, accuracy] = out

		epoch_accuracy = round(np.sum(accuracy[epoch*NoofImages//batchSize:])/(t+1),2)
		
		by = float(t+1)/len(batches)*100

		print("Epoch:"+str(round(by,2))+"% Of "+str(epoch+1)+"/"+str(epochs)+", Cost:"+str(cost[-1])+", Batch.Acc:"+str(accuracy[-1]*100)+", Epoch.Acc:"+str(epoch_accuracy))

		t+=1


## saving the trained model parameters
with open(pickle_file, 'wb') as file:
	pickle.dump(out, file)

## Opening the saved model parameter
pickle_in = open(pickle_file, 'rb')
out = pickle.load(pickle_in)

[filt1, filt2, bias1, bias2, theta, thetaBias, cost, acc] = out

#timePlot = np.arange(epochs*(train_data.shape[0]//batchSize))
#plt.plot(timePlot, CostTrack)
#plt.xlabel('No of Batches')
#plt.ylabel('Mean Accuracy')
#plt.title('Training Batch Accuracy')
 
# function to show the plot
#plt.show()

## Computing Test accuracy
X = test_data[:,0:-1]
X = X.reshape(len(test_data), ImageDepth, ImageWidth, ImageWidth)
y = test_data[:,-1]
count = 0
for i in range(0,len(test_data)):
	image = X[i]
	label, prob = predict(image, filt1, filt2, bias1, bias2, theta, thetaBias)
	if label==y[i]:
		count+=1
test_accuracy = float(count)/len(test_data)*100
print("Test Accuracy:"+str(test_accuracy))




