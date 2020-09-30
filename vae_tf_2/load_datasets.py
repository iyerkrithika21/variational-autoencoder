import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def make_spiral_data(n_samples = 500,n_loops=3,noise_level=0.0):
    '''
    noise : float, (default=0.0)
    The standard deviation of the gaussian noise applied to the output.
    
    n_samples : int, (default=500)
    The number of samples.

    n_loops: int , (default=3)
    The number of loops
    '''

    a = 1.5
    b = -2.4
    t = np.linspace(start = 0,stop=n_loops*np.pi, num=n_samples)
    x = (a + b*t) * np.cos(t)
    y = (a + b*t) * np.sin(t)
    data = np.zeros((n_samples,2))
    noise = np.random.normal(0,noise_level,(n_samples,2))
    data[:,0] = x + noise[:,0]
    data[:,1] = y + noise[:,1]
    return (data,np.zeros(n_samples))



def preprocess_images(images):
  images = images / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')


def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_size = 60000
    batch_size = batch_size
    test_size = 10000
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
    
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size))

    
    optimizer =  tf.keras.optimizers.Adam(1e-4)
    num_examples_to_generate = 16
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :]

    return train_dataset,test_dataset,optimizer,test_sample



def load_toy_data(dataset_name,n_samples,noise_level,batch_size,n_loops,normalize_flag):
    scaler = MinMaxScaler()
    
    if(dataset_name=="moon"):
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise_level,random_state=42)
        
        if(normalize_flag==True):
            X = scaler.fit_transform(noisy_moons[0])
        else:
            X = noisy_moons[0]
        y = noisy_moons[1]
        

    elif(dataset_name=="circles"):

        noisy_cirles = datasets.make_circles(n_samples = n_samples*2,noise = noise_level,random_state=42)

        if(normalize_flag==True):
            X = scaler.fit_transform(noisy_cirles[0])
        else:
            X = noisy_cirles[0]

        X = X[noisy_cirles[1]==0,:]
        y = np.zeros(len(X))


    elif(dataset_name=="spiral"):

        noisy_spiral = make_spiral_data(n_samples,n_loops,noise_level)
        if(normalize_flag==True):
            X = scaler.fit_transform(noisy_spiral[0])
        else:
            X = noisy_spiral[0]

        y = noisy_spiral[1]

    X_train, X_test,_,_ = train_test_split(X,y, test_size=0.4, random_state=42)
    
    train_size = len(X_train)
    batch_size = batch_size
    test_size = len(X_test)

    

    train_dataset = (tf.data.Dataset.from_tensor_slices(X_train.astype('float32')).shuffle(train_size).batch(batch_size))
    
    test_dataset = (tf.data.Dataset.from_tensor_slices(X_test.astype('float32')).shuffle(test_size).batch(batch_size))

    optimizer =  tf.keras.optimizers.Adam(1e-4)
    num_examples_to_generate = len(X_test)
    
    plt.subplot(1,2,1)
    plt.plot(X_train[:,0],X_train[:,1],'o')
    plt.subplot(1,2,2)
    plt.plot(X_test[:,0],X_test[:,1],'o')
    plt.savefig(dataset_name+"_train_test.png")
    plt.close()

    return train_dataset,test_dataset,optimizer,X_train,X_test,scaler