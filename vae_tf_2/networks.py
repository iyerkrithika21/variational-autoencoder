import tensorflow as tf

#from tf.keras import layers


def fc_mnist_encoder(x, latent_dim):
    """
    Inference network q(z|x) which encodes a mini batch of data points
    to a parameterization of a diagonal Gaussian using a network with 
    fully connected layers.

    :param x: Mini batch of data points to encode.
    :param latent_dim: dimension of latent space into which we encode
    :return: e: Encoded mini batch.
    """
    encoder = tf.keras.Sequential(
    	[
    	tf.keras.InputLayer(input_shape=(None,28*28)),
    	tf.keras.layers.Dense(500),
    	tf.keras.layers.Dense(500),
    	tf.keras.layers.Dense(200),
    	tf.keras.layers.Dense(2 * latent_dim),
    	])

    return encoder


def fc_mnist_decoder(z,latent_dim):
    """
    Generative network p(x|z) which decodes a sample z from
    the latent space using a network with fully connected layers.
    
    :param z: Latent variable sampled from latent space.
    :return: x: Decoded latent variable.
    """
    decoder = tf.keras.Sequential(
    	[
    	tf.keras.InputLayer(input_shape=(None,latent_dim)),
    	tf.keras.layers.Dense(200),
    	tf.keras.layers.Dense(500),
    	tf.keras.layers.Dense(500),
    	tf.keras.layers.Dense(units = 28*28,activation = tf.nn.sigmoid),
    	])

    return decoder


def conv_mnist_encoder(x, latent_dim):
    """
    Inference network q(z|x) which encodes a mini batch of data points
    to a parameterization of a diagonal Gaussian using a network with 
    convolutional layers.

    :param x: Mini batch of data points to encode.
    :param latent_dim: dimension of latent space into which we encode
    :return: e: Encoded mini batch.
    """

    encoder = tf.keras.Sequential(
    	[
    	tf.keras.InputLayer(input_shape=(None,28,28,1)),
    	tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(2, 2)),
    	tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2)),
    	tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=(2, 2)),
    	tf.keras.layers.Flatten(),
    	f.keras.layers.Dense(500),
    	tf.keras.layers.Dense(200),
    	tf.keras.layers.Dense(2 * latent_dim),])

    return encoder


def conv_mnist_decoder(z):
    """
    Generative network p(x|z) which decodes a sample z from
    the latent space using a network with convolutional layers.
    
    :param z: Latent variable sampled from latent space.
    :return: x: Decoded latent variable.
    """
    self.decoder = tf.keras.Sequential(
        [
        tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='valid'),
    	tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='valid'),
    	tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=2),
    	tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2,activation=tf.nn.sigmoid),
    	tf.keras.layers.Flatten(),
    	])

    return decoder
