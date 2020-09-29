import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

@tf.function
def train(model,x,optimizer,batch_size,dataset_name):

    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x,batch_size,dataset_name) 
    gradients = tape.gradient(loss, model.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss





def compute_loss(model, x,batch_size,dataset_name):

    mean, logvar,_ = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    if(dataset_name=="mnist"):
        
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit[:,:,:,0], labels=x)
        logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2])
        
        
    elif(dataset_name=="moon" or dataset_name =="circles"):
        cross_ent = tf.keras.losses.MSE(x_logit, x)
        # logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
        logpx_z = cross_ent
        


    # logpz = log_normal_pdf(z, 0., 0.)
    # logqz_x = log_normal_pdf(z, mean, logvar)
    kl_loss = kl_diagnormal_stdnormal(mean,logvar)
    # return tf.reduce_mean(logpx_z + logpz - logqz_x)
    return tf.reduce_mean(kl_loss) + tf.reduce_mean(logpx_z)



def kl_diagnormal_stdnormal(mu, log_var, eps=1e-8):
    """
    Calculates KL Divergence between q~N(mu, sigma^T * I) and p~N(0, I).
    q(z|x) is the approximate posterior over the latent variable z,
    and p(z) is the prior on z.

    :param mu: Mean of z under approximate posterior.
    :param sigma: Standard deviation of z
        under approximate posterior.
    :param eps: Small value to prevent log(0).
    :return: kl: KL Divergence between q(z|x) and p(z).
    """
   
    kl = 1 + log_var - mu**2 - tf.exp(log_var)
    kl = tf.reduce_sum(kl,axis=-1)
    kl = kl*(-0.5)
    kl = tf.reduce_mean(kl)

    return kl




def generate_and_save_images(model, epoch, test_sample):
    mean, logvar,_ = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        pred = predictions[i,:]
        pred = tf.reshape(pred,(1,28,28,1))
        plt.imshow(pred[0, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('mnist/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def generate_and_save_moon(model,epoch,test_sample,scaler,dataset_name):
    mean, logvar,_ = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    #predictions = scaler.inverse_transform(predictions)
    fig = plt.figure(figsize=(4, 4))
    plt.plot(predictions[:,0],predictions[:,1],'o')
    plt.savefig(dataset_name+'/at_epoch_{:04d}.png'.format(epoch))
    plt.close()
        

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)