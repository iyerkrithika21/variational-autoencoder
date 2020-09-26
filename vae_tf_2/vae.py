import numpy as np
import tensorflow as tf
from networks import *



class VAE(tf.keras.Model):

    def __init__(self, latent_dim, batch_size,encoder_fn,decoder_fn):
        """
        Implementation of Variational Autoencoder (VAE) for  MNIST.
        Paper (Kingma & Welling): https://arxiv.org/abs/1312.6114.

        :param latent_dim: Dimension of latent space.
        :param batch_size: Number of data points per mini batch.
        :param encoder: function which encodes a batch of inputs to a 
            parameterization of a diagonal Gaussian
        :param decoder: function which decodes a batch of samples from 
            the latent space and returns the corresponding batch of images.
        """
        super(VAE,self).__init__()
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self.encoder = encoder_fn(self._latent_dim)
        self.decoder = decoder_fn(self._latent_dim)
        
        """
        Build tensorflow computational graph for VAE.
        x -> encode(x) -> latent parameterization & KL divergence ->
        z -> decode(z) -> distribution over x -> log likelihood ->
        total loss -> train step
        """

    @tf.function
    def sample(self,eps=None):
        if eps is None:
            eps = eps = tf.random.normal(shape=(self._batch_size, self.latent_dim))
        return self.decode(eps)

    def encode(self,x):

        encoded = self.encoder(x)
        
        # extract mean and (diagonal) log variance of latent variable
        mean = encoded[:, :self._latent_dim]
        logvar = encoded[:, self._latent_dim:]
        # also calculate standard deviation for practical use
        #stddev = tf.sqrt(tf.exp(logvar))
        stddev = logvar
        return mean, logvar,stddev

    def reparameterize(self,mean,logvar):

        # sample from latent space
        epsilon = tf.random.normal(shape=mean.shape)
        z = mean + tf.exp(logvar * .5) * epsilon

        return z

    def decode(self,z):
            
        decoded = self.decoder(z)
        return decoded

    