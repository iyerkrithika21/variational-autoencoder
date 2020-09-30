import networks as nets
import tensorflow as tf

from plot_utilities import *
# import tensorflow_datasets as tfds
from tqdm import tqdm
from vae import VAE
from load_datasets import *
from training import *
import datetime


def main():
    flags = tf.compat.v1.flags

    # VAE params
    flags.DEFINE_integer("latent_dim", 2, "Dimension of latent space.")
    flags.DEFINE_integer("batch_size", 32, "Batch size.")
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
        'encoders': 
        {
            'mnist':
            {
                'fc': nets.fc_mnist_encoder,
                'conv': nets.conv_mnist_encoder
            },
            "moon":
            {
                "fc":nets.fc_moon_encoder
            },
            "circles":
            {
                'fc':nets.fc_moon_encoder
            },
            "spiral":
            {
                'fc':nets.fc_moon_encoder
            }
        },
        'decoders': 
        {
            'mnist':
            {
                'fc': nets.fc_mnist_decoder,
                'conv': nets.conv_mnist_decoder
            },
            "moon":
            {
                'fc':nets.fc_moon_decoder
            },
            "circles":
            {
                'fc':nets.fc_moon_decoder
            },
            "spiral":
            {
                'fc':nets.fc_moon_decoder
            }

        }
    }

    # define model
    dataset_choice ="spiral"
    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'batch_size': FLAGS.batch_size,
        'encoder_fn': architectures['encoders'][dataset_choice][FLAGS.encoder_architecture],
        'decoder_fn': architectures['decoders'][dataset_choice][FLAGS.decoder_architecture]
    }


    vae = VAE(**kwargs)
    n_loops = 3
    if(dataset_choice=="mnist"):
        train_dataset,test_dataset,optimizer,test_sample = load_mnist_data()
    else:
        train_dataset,test_dataset,optimizer,train_sample,test_sample,scaler = load_toy_data(dataset_choice,2000,0.01,FLAGS.batch_size,n_loops*2,False)


    # Tensorboard log locations
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + dataset_choice +'_'+current_time+ '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    





    #FLAGS.updates_per_epoch = int(data.get_length()/FLAGS.batch_size)
    # do training
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        training_loss = 0.
        
        # iterate through batches
        for train_x in train_dataset:

            #train_x = data.next_batch()

            loss = train(vae,train_x,optimizer,FLAGS.batch_size,dataset_choice)
            training_loss += loss
        
        # average loss over most recent epoch
        training_loss /= (FLAGS.updates_per_epoch)
        # update progress bar
        s = "Loss: {:.4f}".format(training_loss)
        tbar.set_description(s)

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(vae, test_x,FLAGS.batch_size,dataset_choice))
        elbo = -loss.result()
        
        print('Epoch: {}, Test set ELBO: {}'
        .format(epoch, elbo))
        generate_and_save_moon(vae, epoch, test_sample,scaler,dataset_choice)
        # generate_and_save_images(vae,epoch,test_sample)
    plot_latent_space_kde(vae,test_sample,train_sample)
    


    #     with train_summary_writer.as_default():
    #         tf.summary.scalar('loss', training_loss, step=epoch)
    #         tf.summary.scalar('Test ELBO',elbo,step=epoch)
            

    
    



    # #     # make pretty pictures if latent dim. is 2-dimensional
    # #     if FLAGS.latent_dim == 2 and FLAGS.do_viz:
    # #         make_canvas(vae=vae, batch_size=FLAGS.batch_size, epoch=epoch)
    # #         make_spread(vae, provider, epoch)



if __name__ == '__main__':
    main()
