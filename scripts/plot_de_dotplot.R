require(ggplot2)
library(scales)

bg = read.table("../ll_bg.txt", sep='\t', header=T, row.names=1, stringsAsFactors=F)
ll29 = read.table("../lg29_ll.txt", sep='\t', header=T, row.names=1, stringsAsFactors = F)
ll36 = read.table("../lg36_ll.txt", sep='\t', header=T, row.names=1, stringsAsFactors = F)
ll78 = read.table("../lg78_ll.txt", sep='\t', header=T, row.names=1, stringsAsFactors = F)
ll94 = read.table("../lg94_ll.txt", sep='\t', header=T, row.names=1, stringsAsFactors = F)

bgs = bg[abs(bg$log2fc) > 1,]
ll29s = ll29[abs(ll29$log2fc) > 1,]
ll36s = ll36[abs(ll36$log2fc) > 1,]
ll78s = ll78[abs(ll78$log2fc) > 1,]
ll94s = ll94[abs(ll94$log2fc) > 1,]

top_genes = c(ll29s[1:10, 'gene'], ll36s[1:10, 'gene'], ll78s[1:10, 'gene'], ll94s[1:10, 'gene'], 'REG4', 'IFI27', 'ASS1')
top_genes = unique(top_genes)

df = data.frame()
j = 1
for (i in top_genes) {
  
  s29 = ll29[ll29$gene == i, 'qval']
  fc29 = ll29[ll29$gene == i, 'log2fc']
  
  s36 = ll36[ll36$gene == i, 'qval']
  fc36 = ll36[ll36$gene == i, 'log2fc']
  
  s78 = ll78[ll78$gene == i, 'qval']
  fc78 = ll78[ll78$gene == i, 'log2fc']
  
  s94 = ll94[ll94$gene == i, 'qval']
  fc94 = ll94[ll94$gene == i, 'log2fc']
  
  sbg = bg[bg$gene == i, 'qval']
  fcbg = bg[bg$gene == i, 'log2fc']
  
  df[j, 'gene'] = i
  df[j, 'clone'] = '29'
  df[j, 'log2fc'] = fc29
  df[j, 'score'] = s29
  j <- j + 1
  
  df[j, 'gene'] = i
  df[j, 'clone'] = '36'
  df[j, 'log2fc'] = fc36
  df[j, 'score'] = s36
  j <- j + 1
  
  df[j, 'gene'] = i
  df[j, 'clone'] = '78'
  df[j, 'log2fc'] = fc78
  df[j, 'score'] = s78
  j <- j + 1
  
  df[j, 'gene'] = i
  df[j, 'clone'] = '94'
  df[j, 'log2fc'] = fc94
  df[j, 'score'] = s94
  j <- j + 1
  
  df[j, 'gene'] = i
  df[j, 'clone'] = 'BG'
  df[j, 'log2fc'] = fcbg
  df[j, 'score'] = sbg
  j <- j + 1
  
  
}

genes.met = df[df$clone == 'BG',][which(df[df$clone == 'BG', 'log2fc'] > 0), 'gene']
genes.dwn = setdiff(unique(df$gene), genes.met)

df$gene <- factor(df$gene, levels=c(genes.met, genes.dwn))

# plot dotplot 
ggplot(df, aes(gene, clone, size = -log10(score), color=log2fc)) + 
  geom_point() + 
  theme_bw() + 
  # scale_colour_gradient2(low="#00AEEF", mid='white', high="red") + 
  scale_colour_gradientn(colours=c('#00AEEF', 'white', 'red'), values  = rescale(c(min(df$log2fc), 0.5)), oob=squish) + 
  theme(axis.text.x = element_text(angle = 90))
ggsave("ll_de_dotplot.eps", scale=1.5)
