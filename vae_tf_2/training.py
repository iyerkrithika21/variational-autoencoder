import tensorflow as tf

@tf.function
def train(model,x,optimizer,batch_size):

    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x,batch_size)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def compute_loss(model,x,batch_size):

    mean,logvar,stddev = model.encode(x)
    z = model.reparameterize(mean,stddev)
    decoded = model.decode(z)

    # calculate KL divergence between approximate posterior q and prior p
    kl = kl_diagnormal_stdnormal(mean, stddev)

    # calculate reconstruction error between decoded sample
    # and original input batch
    log_like = bernoulli_log_likelihood(x, decoded)

    loss = (kl + log_like) / batch_size
    return loss

def kl_diagnormal_stdnormal(mu, sigma, eps=1e-8):
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
    var = tf.square(sigma)
    kl = 0.5 * tf.reduce_sum(input_tensor=tf.square(mu) + var - 1. - tf.math.log(var + eps))
    return kl


def bernoulli_log_likelihood(targets, outputs, eps=1e-8):
    """
    Calculates negative log likelihood -log(p(x|z)) of outputs,
    assuming a Bernoulli distribution.

    :param targets: MNIST images.
    :param outputs: Probability distribution over outputs.
    :return: log_like: -log(p(x|z)) (negative log likelihood)
    """
    log_like = -tf.reduce_sum(input_tensor=targets * tf.math.log(outputs + eps)
                              + (1. - targets) * tf.math.log((1. - outputs) + eps))
    return log_like


