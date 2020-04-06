require(ggplot2)
require(ggrepel)

args = commandArgs(trailingOnly = T)
fp = args[[1]]
title = args[[2]]
save_fp = args[[3]]

genes = read.table(fp, sep='\t', header=T, row.names=1)
genes$log10qval = -log10(genes$fdr)
genes$gene = rownames(genes)

# remove genes whose pvalue is 0
genes[is.infinite(genes$log10qval),"log10qval"] = 300
genes$log10qval = unlist(lapply(genes$log10qval, function(x) min(300, x)))
#genes = genes[is.finite(genes$log10_qval),]

print(genes[genes$gene == "IFI6",])

hits = genes[genes$thresh == 'True', ]
hit.genes = hits$gene

pos = hits[hits$betas >= 0, ]
neg = hits[hits$betas <= 0, ]

pos.ordered = pos[order(-pos$betas), ]
neg.ordered = neg[order(neg$betas), ]

pos.genes = pos.ordered$gene[1:15]
neg.genes = neg.ordered$gene[1:15]

write.table(pos.ordered, 'poisson_de.poshits.txt', sep='\t')
write.table(neg.ordered, 'poisson_de.neghits.txt', sep='\t')

# pos.quantile = quantile(pos$log10qval[is.finite(pos$log10qval)], c(0.5))[[1]]
# neg.quantile = quantile(neg$log10qval[is.finite((neg$log10qval))], c(0.5))[[1]]

# pos.genes = as.character(pos[pos$log10qval >= pos.quantile,"gene"])
# neg.genes = as.character(neg[neg$log10qval >= neg.quantile,"gene"])

# annot.genes = c('ID3', 'IFI6', 'REG4', 'S100A6', 'S100A4', 'CCL20', 'DUSP1', 'CLU', 'AGR3', 
#                'IFGBP3', 'PRKCDBP', 'NEAT1', 'MALAT1', 'FXYD2', 'IGFBP6', 'NFKBIA', 'IFI27', 
#                'B2M', 'KRT17', 'KRT19', 'ID1', 'LGALS1', 'RPS4Y1')

annot.genes = c('REG4', 'IFI27', 'PRKCDBP', 'IFI6', 'TNNT1', 
                'KRT17', 'RPS4Y1', 'FXYD2', 'ID3', 'ASS1')

# annot.genes = c(pos.genes, neg.genes)

ggplot(genes, aes(log2fc, betas, label=gene)) + geom_point(data = subset(genes, thresh == 'True' & log2fc >= 0), color='red') + 
  geom_point(data = subset(genes, thresh == 'False'), color='black') + 
  geom_point(data = subset(genes, thresh == 'True' & log2fc < 0), color='blue') + 
  geom_text_repel(data = subset(genes, gene %in% hit.genes), size = 2.5) + 
  labs(x = 'Log2FC', y = 'Beta', title = 'Poisson DE vs Log2FC') +
  theme_classic() + theme(aspect.ratio = 1)

# ggplot(genes, aes(log2fc, log10qval, label=gene)) + geom_point() + 
#   geom_point(data = subset(genes, gene %in% hit.genes), color = 'red') + 
#  geom_text_repel(data = subset(genes, gene %in% annot.genes)) + 
#  xlim(-2.5, 2.5) +
#  theme_classic() + theme(text=element_text(size=16,  family="Comic Sans MS"))

ggsave("poisson_de.lg.eps")

# ggplot(genes, aes(log2fc, log10_qval, label=gene)) + geom_point() + 
#   geom_hline(yintercept = -log10(0.05), color="red", linetype = "dotted") + 
#   geom_vline(xintercept = -0.70, color="red", linetype = "dotted") + 
#   geom_vline(xintercept = 0.70, color="red", linetype = "dotted") + 
#   geom_text_repel(data = subset(genes, gene %in% keep.genes)) + 
#   geom_point(color = ifelse(genes$gene %in% keep.genes, "red", "black"), size = ifelse(genes$gene %in% keep.genes, 1, 1/100)) +
#   xlim(-3, 3) + 
#   ggtitle(title)

# ggsave(save_fp)
