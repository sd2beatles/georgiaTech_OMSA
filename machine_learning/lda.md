![image](https://user-images.githubusercontent.com/53164959/76708301-17e67f00-6739-11ea-9286-c5c09ab351cf.png)

![image](https://user-images.githubusercontent.com/53164959/76708309-2cc31280-6739-11ea-867e-48f958be9fe3.png)

```python
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
import seaborn as sns

def lda(X,y,redDim):
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    #frist,standardize the data
    X=(X-mean)/std
    #define the number of instances or samples
    ndata=np.shape(X)[0]
    #define the number of dimenions in the datset
    ndim=np.shape(X)[1]
    assert redDim<=ndim,"the number of selected dimension should not be greater than the dimension of data"
    #variation within the classes
    Sw=np.zeros((ndim,ndim))
    sb=np.zeros((ndim,ndim))
    
    #compute the totoal vairation in the dataset by implmenting the np.cov
    #make sure that the input must be structured in a shape like (ndim,ndata)
    #we should tranpose the original data
    St=np.cov(np.transpose(X))
    
    #to identify the classes in our data
    classes=np.unique(y)
    for i in range(len(classes)):
        #np.where returns (1,number of relevant indices belonging to each class)
        indices=np.where(y==classes[i])
        #but (1,number) can not be used for indexing. Only(number,) is applicable 
        #We can do this by implmenting np.squeeze which removes single dimensional entires from the shape of an array
        indices=np.squeeze(indices)
        #find the relevant instances belong to class i in our dataset
        d=X[indices,:]
        #now we are ready to compute the covariance for each class
        classcov=np.cov(np.transpose(d))
        #Bear in mind that we should add the covariance of each class to Sw.But the only difference here is that
        #we add the classcov according to its proportion of number of instances of each class to the total number of samples
        Sw+=np.float(np.shape(indices)[0])/ndata*classcov
    #Variation between classes is easily calculated by Sb=St-Sw
    Sb=St-Sw
    #compute the eigenvalues,eigenvecors and sort them into order
    #scipy.linal.eig(a,b)
    #a:A complex or real matrix whose eigenvalues and eigenvectors will be computed.
    #b:Right-hand side matrix in a generalized eigenvalue problem. Default is None, identity matrix is assumed.
    evals,evecs=la.eig(Sw,Sb)
    indices=np.argsort(evals)
    #note that the indicies are now rearraged in a descending order
    indices=indices[::-1]
    #find the linear lines that maximizes the variation between classes but minizes the variation within each class
    w=evecs[:,:redDim] 
    #now newData is every single data projected onto the line
    newData=np.dot(X,w)
    return newData,w
newData,w=lda(data,labels,2)
sns.set()
plt.title("LDA : Before and After Transformation")
plt.plot(data[:,0],data[:,1],'ro',newData[:,0],newData[:,0],'b.')
plt.show()
```
