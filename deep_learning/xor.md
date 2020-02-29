# 1.Introduction 

xor case had been considered a stickingpoint not until nueral net was finally invented to solve the problem. In this tutorial, we are
going to implement a simple two-layer neural net to solve xor problem

![image](https://user-images.githubusercontent.com/53164959/75609417-b1b00880-5b4b-11ea-866e-399da80c562b.png)


# 2. code


```python

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
tfe=tf.contrib.eager
tf.set_random_seed(777)  # for reproducibility
tf.executing_eagerly()

x_data=[[0,0],
        [0,1],
        [1,0],
        [1,1]]
y_data=[[0],[1],[1],[0]]


data=tf.data.Dataset.from_tensor_slices((x_data,y_data)).batch(len(x_data))
def preprocessing_data(features,labels):
  features=tf.cast(features,tf.float32)
  labels=tf.cast(labels,tf.float32)
  return features,labels

W1=tf.Variable(tf.random_normal([2,1]),name='weight1')
b1=tf.Variable(tf.random_normal([1]),name='bias1')

W2=tf.Variable(tf.random_normal([2,1]),name='weight2')
b2=tf.Variable(tf.random_normal([1]),name='bias2')

W3=tf.Variable(tf.random_normal([2,1]),name="Weight3")
b3=tf.Variable(tf.random_normal([1]),name="bias3")

def neural_net(features):
  layer1=tf.sigmoid(tf.matmul(features,W1)+b1)
  layer2=tf.sigmoid(tf.matmul(features,W2)+b2)
  #concatenate tensors from layer1 and layer 2  along vertical axis
  layer3=tf.concat([layer1,layer2],axis=1)
  #double check if two tensors are mergered safely
  layer3=tf.reshape(layer3,shape=[-1,2])
  hypothesis=tf.sigmoid(tf.matmul(layer3,W3)+b3)
  #hypothesis returns the vector with (4,1) dimension
  return hypothesis

def loss_fn(hypothesis,labels):
  cost=-tf.reduce_mean(labels*tf.log(hypothesis)+(1-labels)*tf.log(1-hypothesis))
  return cost

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)

def accuracy_fn(hypothesis,labels):
  predicted=tf.cast(hypothesis>0.5,dtype=tf.float32)
  accuracy=tf.reduce_mean(tf.equal(predicted,lables),dtype=tf.float32)*100
  return accuracy

def grad(features,labels):
  with tf.GradientTape() as tape:
    loss_value=loss_fn(neural_net(features),labels)
  return tape.gradient(loss_value,[W1,W2,W3,b1,b2,b3])

epoch=50000
for step in range(epoch):
  for features,labels in tfe.Iterator(data):
    features,labels=preprocessing_data(features,labels)
    grads=grad(features,labels)
    optimizer.apply_gradients(grads_and_vars=zip(grads,[W1,W2,W3,b1,b2,b3]))
    if step%500==0:
      print(f"iter{step}, loss:{loss_fn(neural_net(features),labels)}")
#after training data is complete, we need to compute the acccuracy of predciting result based on the trained model
x_data,y_data=preprocessing_data(x_data,y_data)
print(f"accuracy:{accuracy_fn(x_data,y_data)}")
```
