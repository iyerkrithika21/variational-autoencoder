import networks as nets
import tensorflow as tf

from plot import make_canvas, make_spread, make_canvas_gif, make_spread_gif
import tensorflow_datasets as tfds
from tqdm import tqdm
from vae import VAE
from datasets import MNISTDataset
from training import *
import datetime


def main():
    flags = tf.compat.v1.flags

    # VAE params
    flags.DEFINE_integer("latent_dim", 2, "Dimension of latent space.")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    # architectures
    flags.DEFINE_string("encoder_architecture", 'fc', "Architecture to use for encoder.")
    flags.DEFINE_string("decoder_architecture", 'fc', "Architecture to use for decoder.")

    # training params
    flags.DEFINE_integer("epochs", 100,
                         "Total number of epochs for which to train the model.")
    flags.DEFINE_integer("updates_per_epoch", 100,
                         "Number of (mini batch) updates performed per epoch.")

    # data params
    flags.DEFINE_string("data_dir", '../mnist', "Directory containing MNIST data.")
    FLAGS = flags.FLAGS

    # viz params
    flags.DEFINE_bool("do_viz", True, "Whether to make visualisations for 2D.")

    architectures = {
        'encoders': {
            'fc': nets.fc_mnist_encoder,
            'conv': nets.conv_mnist_encoder
        },
        'decoders': {
            'fc': nets.fc_mnist_decoder,
            'conv': nets.conv_mnist_decoder
        }
    }

    # define model
    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'batch_size': FLAGS.batch_size,
        'encoder_fn': architectures['encoders'][FLAGS.encoder_architecture],
        'decoder_fn': architectures['decoders'][FLAGS.decoder_architecture]
    }
    vae = VAE(**kwargs)

    # data provider
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()



    data = MNISTDataset(train_images.reshape([-1, 784]), train_labels, 
                    test_images.reshape([-1, 784]), test_labels,
                    batch_size=FLAGS.batch_size)
    optimizer =  tf.keras.optimizers.Adam(1e-4)


    # Tensorboard log locations
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    





    # do training
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        training_loss = 0.

        # iterate through batches
        for _ in range(FLAGS.updates_per_epoch):
            train_x, _ = data.next_batch()
            loss = train(vae,train_x,optimizer,FLAGS.batch_size)
            training_loss += loss

        # average loss over most recent epoch
        training_loss /= (FLAGS.updates_per_epoch)
        # update progress bar
        s = "Loss: {:.4f}".format(training_loss)
        tbar.set_description(s)


        with train_summary_writer.as_default():
            tf.summary.scalar('loss', training_loss, step=epoch)
            tf.summary.scalar('accuracy', training_loss, step=epoch)





    #     # make pretty pictures if latent dim. is 2-dimensional
    #     if FLAGS.latent_dim == 2 and FLAGS.do_viz:
    #         make_canvas(vae=vae, batch_size=FLAGS.batch_size, epoch=epoch)
    #         make_spread(vae, provider, epoch)

    # # make
    # if FLAGS.latent_dim == 2 and FLAGS.do_viz:
    #     make_canvas_gif()
    #     make_spread_gif()


if __name__ == '__main__':
    main()
