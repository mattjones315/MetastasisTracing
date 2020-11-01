require(VISION)
require(Matrix)

options(mc.cores = 10)


proj_dir = "/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/ALL_Samples/GRCh38/"

data = readMM(paste0(proj_dir, "matrix.mtx"))
genes = read.table(paste0(proj_dir, "genes.tsv"), stringsAsFactors=F)[,2]
bc = read.table(paste0(proj_dir, "barcodes.tsv"), sep='\t', stringsAsFactors = F)[,1]

colnames(data) <- bc
rownames(data) <- genes

norm.factor = median(colSums(data))
data <- t( t(data) / colSums(data)) * norm.factor

#f.genes = VISION:::filterGenesFano(data)
f.genes = read.table("filtered_genes.txt", sep='\t', header=F, stringsAsFactors=F)[,1]
print(head(f.genes))
print(length(f.genes))

meta = read.table("/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/LG_meta.txt", sep='\t', header=T, row.names=1, na.strings=c("NA", ""))

meta$LineageGroup = as.factor(meta$LineageGroup)
meta$Sample2 = as.factor(meta$Sample2)
meta$sampleID = as.factor(meta$sampleID)
meta = meta[-which(is.na(meta$scTreeMetRate)),]

# subsample to only those in the TS library
kii = intersect(bc, rownames(meta))
data = data[,kii]
meta = meta[kii,]

meta$nUMI = colSums(data)
meta$nGenes = colSums(data > 0)

latent = read.table("latent.csv", sep='\t', header=T, row.names=1)
rownames(latent) = bc
latent = latent[kii,]

latent = latent[rownames(meta),]
data = data[,rownames(meta)]

print(dim(data))

sigs = c("/data/yosef2/users/mattjones/data/h.all.v5.2.symbols.gmt", "/data/yosef2/users/mattjones/data/c2.all.v6.0.symbols.gmt", "/data/yosef2/users/mattjones/data/c6.all.v6.0.symbols.gmt")

vis <- Vision(data, sigs, projection_genes = f.genes, meta=meta, pool=F, latentSpace=latent, projection_methods=c("tSNE30", "UMAP"))
vis <- analyze(vis)

saveRDS(vis, "5k_ALL_vision.rds")
sigScores = vis@SigScores

write.table(sigScores, '/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/JQ19.sigscores.txt', sep='\t')

viewResults(vis, host='0.0.0.0', port=8224, browser=F)
