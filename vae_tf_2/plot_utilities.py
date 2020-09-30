import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow.keras.backend as K
import numpy as np

def plot_latent_space_kde(model,test_sample,train_sample):
	
	mean, logvar,_ = model.encode(test_sample)
	z = model.reparameterize(mean, logvar)
	x_logit = model.decode(z)
	x = K.get_value(x_logit)
	z = K.get_value(z)

	plt.rcParams.update({'font.size': 22})
	plt.figure(figsize =(40,20))
	plt.subplot(1,3,1)
	sns.kdeplot(z[:, 0], z[:, 1], shade=True, cmap='Blues', bw=0.3)
	plt.title("Test sample latent space")

	plt.subplot(1,3,2)
	plt.plot(test_sample[:,0],test_sample[:,1],'co')
	plt.plot(x[:,0],x[:,1],'mo')
	plt.legend(['Test sample','Test recon'])
	plt.title("Test samples and the reconstruction")

	
	mean, logvar,_ = model.encode(train_sample)
	z = model.reparameterize(mean, logvar)
	x_logit = model.decode(z)
	x = K.get_value(x_logit)
	z = K.get_value(z)




	plt.subplot(1,3,3)
	plt.plot(train_sample[:,0],train_sample[:,1],'co')
	plt.plot(x[:,0],x[:,1],'mo')
	plt.legend(['Train sample','Train recon'])
	plt.title("Train samples and the reconstruction")

	plt.savefig("test_latent_recon.png")
	plt.close()