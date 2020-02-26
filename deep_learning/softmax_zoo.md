# 1. Data

You can visit the fllowing the website to download the data named 'data-04-zoo.csv'. After the data is prepared, let's get to the next
step

[https://github.com/hunkim/DeepLearningZeroToAll/blob/master/data-04-zoo.csv]

# 2.Brief Intrdouction

The data consists of 101 rows and 17 columns and each input fearues desribe the distict features of animlas except the last column which
shows the group it belongs to. Please note that the whole group ranges from 0 to 6 which amounts to 7 in total. Since our ultimate goal is
to find the proper group given the input data. The number of labels are more than two, thereby sigmoid function being not applicable as
the activation function. Let's transform the values to represetnt which class to predict out of the 7 classes?

# 3. Codes


```python

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
tf.set_random_seed(777)  # for reproducibility
tfe = tf.contrib.eager
from google.colab import files
uploaded = files.upload()
import io

xy=np.loadtxt(io.BytesIO(uploaded['data-04-zoo.csv']),delimiter=',',dtype=np.float32)
print(xy.shape)
x_data = xy[:, 0:-1]
y_data = xy[:, -1].astype(np.int32)

nb_classes=7
#make the  category variable into dummy variables
y_one_hot=tf.one_hot(y_data,nb_classes)
y_one_hot=tf.reshape(y_one_hot,[-1,nb_classes])

#Set up the weight and bias 
#Make sure that you need to match the number of X colums to rows of the weight vector.
W=tfe.Variable(tf.random_normal([16,nb_classes]),name='weight')
#bias is the vector consisting of 16 constant values.
b=tfe.Variable(tf.random_normal([nb_classes]),name='bias')
variables = [W, b]

def hypothesis(X):
  logit_fn=tf.matmul(X,W)+b
  return tf.nn.softmax(logit_fn)

def cost_fn(X,y):
  hypoth=hypothesis(X)
  #cost returns  the sum value in amount to 101. We need to average them out. 
  cost=-tf.reduce_sum(y*tf.log(hypoth),axis=1)
  cost=tf.reduce_mean(cost)
  return cost

def grad_fn(X,y):
  #now ready to take a gradient descendent to find the prameters with cost being minimized
  with tf.GradientTape() as tape:
    loss=cost_fn(X,y)
    grads=tape.gradient(loss,variables)
    return grads
  
def prediction(X,y):
  #return the index whose hypoetheis value is highest and iterates over for 101 times.
  pred=tf.argmax(hypothesis(X),axis=1)
  #to check out whether our prediction matches to the actual results
  correct_prediction=tf.equal(pred,tf.argmax(y,axis=1))
  
  #tf.cast converts boolean value to either 0 or 1. If true, it returns 1 or 0 for false
  accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100
  return accuracy

def fit(X,y,epochs=1000,verbose=100):
  optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
  for i in range(epochs):
    grads=grad_fn(X,y)
    optimizer.apply_gradients(zip(grads,variables))
    if (i==0) | ((i+1)%verbose==0):
      acc=prediction(X,y).numpy()
      loss=cost_fn(X,y).numpy()
      print('Steps: {} Loss: {}, Acc: {}'.format(i+1, loss, acc))
    
fit(x_data, y_one_hot)

```

