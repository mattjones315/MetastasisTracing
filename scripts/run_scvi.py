from scvi.dataset import Dataset10X
from scvi.dataset import AnnDatasetFromAnnData
import scanpy as sc
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(10)

print("Reading in data...")
path = '/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/ALL_Samples/GRCh38/'
adata = sc.read(path + "matrix.mtx", cache=True).T
genes = pd.read_csv(path + "genes.tsv", header=None, sep='\t')
adata.var_names = genes[1]
adata.var['gene_ids'] = genes[0]  # add the gene ids as annotation of the variables/genes
adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]
adata.var_names_make_unique()

#data = Dataset10X(save_path="GRCh38", measurement_names_column = 1)
gene_list = pd.read_csv("filtered_genes.txt", sep='\t', header=None).iloc[:,0].values

#data.filter_genes(gene_list, on="gene_symbols")
filter_result = list(map(lambda x: x in gene_list, adata.var_names))
adata = adata[:, filter_result]

data = AnnDatasetFromAnnData(adata)

n_epochs=40
lr = 1e-3
use_batches = False
use_cuda = True

print("Instantiating VAE...")
vae = VAE(data.nb_genes, n_batch=data.n_batches * use_batches)
trainer = UnsupervisedTrainer(vae, data, train_size = 0.9, use_cuda=use_cuda, frequency=1)
trainer.train(n_epochs=n_epochs, lr=lr)

print("Plotting...")
h = plt.figure(figsize = (10,10))
elbo_train = trainer.history["elbo_train_set"]
elbo_test = trainer.history["elbo_test_set"]
x = np.linspace(0, n_epochs, (len(elbo_train)))
plt.plot(x, elbo_train, label="train")
plt.plot(x, elbo_test, label="test")
# plt.ylim(min(elbo_train)-50, 1000)
plt.legend()
plt.savefig("losscurve.png")

print("Saving latent space...")
posterior = trainer.create_posterior(
                trainer.model, data, indices=np.arange(len(data))
                    ).sequential()
latent, _, _ = posterior.get_latent()
pd.DataFrame(latent).to_csv("latent.csv", sep='\t')
