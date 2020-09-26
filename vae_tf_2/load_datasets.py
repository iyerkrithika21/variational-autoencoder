import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# class MyDataset:
#     """'Bare minimum' class to wrap MNIST numpy arrays into a dataset."""
#     def __init__(self, train_imgs, test_imgs, batch_size, 
#         shuffle=True, seed=None):
#         """
#         Use seed optionally to always get the same shuffling (-> reproducible
#         results).
#         """
#         self.batch_size = batch_size
#         self.train_data = train_imgs
#         self.test_data = test_imgs    
        

#         self.size = self.train_data.shape[0]

#         if seed:
#             np.random.seed(seed)
#         if shuffle:
#             self.shuffle_train()
#         self.shuffle = shuffle
#         self.current_pos = 0

#     def next_batch(self):
#         """Either gets the next batch, or optionally shuffles and starts a
#         new epoch."""
#         end_pos = self.current_pos + self.batch_size
#         if end_pos < self.size:
#             batch = (self.train_data[self.current_pos:end_pos])
#             self.current_pos += self.batch_size
#         else:
#             # we return what's left (-> possibly smaller batch!) and prepare
#             # the start of a new epoch
#             batch = (self.train_data[self.current_pos:self.size])
#             if self.shuffle:
#                 self.shuffle_train()
#             self.current_pos = 0
#             print("Starting new epoch...")
#         return batch

#     def shuffle_train(self):
#         shuffled_inds = np.arange(self.train_data.shape[0])
#         np.random.shuffle(shuffled_inds)
#         self.train_data = self.train_data[shuffled_inds]

#     def get_length(self):
#         return len(self.train_data)
        




# def preprocess_images(images):
#   images = images / 255.
#   return np.where(images > .5, 1.0, 0.0).astype('float32')




def load_data_and_optimiser(dataset_name,n_samples,noise_level,batch_size):
    scaler = MinMaxScaler()
    if(dataset_name=="mnist"):
        # data provider
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

        return train_dataset,test_dataset,test_sample,num_examples_to_generate,scaler


    elif(dataset_name=="moon"):


        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise_level,random_state=42)
        
        
        X = scaler.fit_transform(noisy_moons[0])
        X_train, X_test,_,_ = train_test_split(X,noisy_moons[1], test_size=0.01, random_state=42)
        # print(X_train.max(),X_train.min())
        train_size = len(X_train)
        batch_size = batch_size
        test_size = len(X_test)
        train_dataset = (tf.data.Dataset.from_tensor_slices(X_train.astype('float32')).shuffle(train_size).batch(batch_size))
        
        test_dataset = (tf.data.Dataset.from_tensor_slices(X_test.astype('float32')).shuffle(test_size).batch(batch_size))

        optimizer =  tf.keras.optimizers.Adam(1e-5)
        num_examples_to_generate = 300
        for test_batch in test_dataset.take(1):
            test_sample = test_batch[0:num_examples_to_generate, :]

        return train_dataset,test_dataset,optimizer,test_sample,num_examples_to_generate,scaler



