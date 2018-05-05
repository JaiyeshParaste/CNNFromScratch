import numpy as np
import pickle
from convnet import *


## Hyperparameters
ImageWidth= 28
ImageDepth = 1
pickle_file = 'trained.pickle'

## Extraction of MNIST Test data
m =10000
X = extract_Images('t10k-images-idx3-ubyte.gz', m, ImageWidth)
y = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))


## Opening the saved model parameter


pickle_in = open(pickle_file, 'rb')
out = pickle.load(pickle_in,encoding='latin1')
[filt1, filt2, bias1, bias2, theta, thetabias, cost, accuracy] = out
count = 0


## Testing the Accuracy
for i in range(20,50):
	image = X[i].reshape(ImageDepth, ImageWidth, ImageWidth)
	label, prob = predict(image, filt1, filt2, bias1, bias2, theta, thetabias)
	if label==y[i]:
		count+=1
test_accuracy = float(count)/980*100
print("Testing Accuracy:"+str(test_accuracy))

