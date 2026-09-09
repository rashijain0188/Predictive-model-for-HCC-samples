#Combatseq Data Preprocessing for Training set

#TCGA-LIHC project in The Cancer Genome Atlas Database contains the data files for TCGA data. 
#Gene Expression Omnibus Database contains the files for GSE144269, GSE207435, and GSE214846.

setwd("...")
mRNA_meta<-  read.csv("TCGA_metadata_.csv", header=TRUE)
mRNA_clinical<-  read.csv("TCGA_clinical_data.csv", header=TRUE)

TCGA_counts2 <- read_delim("TCGA_counts.csv", ",")

GSE144269_counts.x <- read_delim("GSE144269_raw_counts_GRCh38.p13_NCBI.tsv","\t")
GSE144269_metadata <- read_delim("Human.GRCh38.p13.annot.tsv","\t")
GSE144269_sample <- read_delim("GSE144269_series_matrix.txt","\t")
levels(as.factor(GSE144269_metadata$GeneType))
GSE144269_counts.x <- GSE144269_counts.x  %>% 
  left_join(GSE144269_metadata[, c(1, 2, 5, 6)], by = "GeneID")
GSE144269_counts.x<- as.data.frame(GSE144269_counts.x)
GSE144269_counts<- GSE144269_counts.x[,2:142]

GSE207435_counts <- read_delim("GSE207435_raw_counts_GRCh38.p13_NCBI.tsv", "\t")
GSE207435_sample <- read_delim("GSE207435_series_matrix.txt","\t")
GSE207435_metadata <- read_delim("Human.GRCh38.p13.annot.tsv","\t")
levels(as.factor(GSE207435_metadata$GeneType))
GSE207435_counts <- GSE207435_counts %>% 
  left_join(GSE207435_metadata[, c(1, 2, 5, 6)], by = "GeneID")
GSE207435_counts<- as.data.frame(GSE207435_counts)
GSE207435_counts<- GSE207435_counts[,2:56]

GSE214846_counts<- read_delim("GSE214846_raw_counts_GRCh38.p13_NCBI.tsv","\t")
GSE214846_metadata <- read_delim("Human.GRCh38.p13.annot.tsv","\t")
GSE214846_sample <- read_delim("GSE214846_series_matrix.txt","\t")
levels(as.factor(GSE214846_metadata$GeneType))
GSE214846_counts <- GSE214846_counts %>% 
  left_join(GSE214846_metadata[, c(1, 2, 5, 6)], by = "GeneID")
GSE214846_counts<- as.data.frame(GSE214846_counts)
GSE214846_counts<- GSE214846_counts[,2:132]

match(TCGA_counts2$gene_name , GSE144269_counts$Symbol) # ALL GEO datasets match

TCGA_counts3 <- TCGA_counts2[TCGA_counts2$gene_name %in% GSE214846_counts$Symbol, ]

common_genes <- Reduce(intersect, list(TCGA_counts3$gene_name, GSE144269_counts3$Symbol, GSE207435_counts3$Symbol, GSE214846_counts3$Symbol))
print(common_genes)

GSE144269_counts <- GSE144269_counts3[GSE144269_counts3$Symbol %in% common_genes, ]
GSE207435_counts <- GSE207435_counts3[GSE207435_counts3$Symbol %in% common_genes, ]
GSE214846_counts <- GSE214846_counts3[GSE214846_counts3$Symbol %in% common_genes, ]
TCGA_counts2  <- TCGA_counts3[TCGA_counts3$gene_name %in% common_genes, ]


# First, make sure all datasets have the same join column name
colnames(TCGA_counts2)[312]<- "Symbol"
rownames(GSE144269_counts)<- GSE144269_counts$Symbol
rownames(GSE214846_counts)<- GSE214846_counts$Symbol
rownames(GSE207435_counts)<- GSE207435_counts$Symbol
rownames(TCGA_counts2)<- TCGA_counts2$Symbol

TCGA_counts2<- as.data.frame(TCGA_counts2)

GSE144269_counts<- GSE144269_counts[-104]
GSE214846_counts<- GSE214846_counts[-88]
GSE207435_counts<- GSE207435_counts[-32]
TCGA_counts2<- TCGA_counts2[,4:305]

GSE144269_counts<- GSE144269_counts[-1]
GSE214846_counts<- GSE214846_counts[-1]
GSE207435_counts<- GSE207435_counts[-1]

GSE144269_counts <- GSE144269_counts[order(rownames(GSE144269_counts)), ]
GSE214846_counts <- GSE214846_counts[order(rownames(GSE214846_counts)), ]
GSE207435_counts <- GSE207435_counts[order(rownames(GSE207435_counts)), ]
TCGA_counts2 <- TCGA_counts2[order(rownames(TCGA_counts2)), ]

merged_counts_raw <- cbind(GSE144269_counts,GSE214846_counts,GSE207435_counts, TCGA_counts2)

GSE144269_sample<- t(GSE144269_sample)
GSE207435_sample<- t(GSE207435_sample)
GSE214846_sample<- t(GSE214846_sample)
GSE207435_sample<- GSE207435_sample[-1,]
GSE214846_sample<- GSE214846_sample[-1,]
GSE144269_sample<- GSE144269_sample[-1,]

GSE207435_sample<- as.data.frame(GSE207435_sample)
GSE214846_sample<- as.data.frame(GSE214846_sample)
GSE144269_sample<- as.data.frame(GSE144269_sample)

colnames(GSE207435_sample)[1]<- "sample_type"
colnames(GSE214846_sample)[1]<- "sample_type"
colnames(GSE144269_sample)[1]<- "sample_type"

GSE144269_sample$Batch <- "GSE144269"
GSE214846_sample$Batch<- "GSE214846"
GSE207435_sample$Batch <- "GSE207435"

GSE144269_sample$sample_type <- gsub("tumor/non-tumor: tumor", "Tumor", GSE144269_sample$sample_type)
GSE144269_sample$sample_type <- gsub("tumor/non-tumor: non-tumor", "Normal", GSE144269_sample$sample_type)
GSE214846_sample$sample_type <- gsub("tissue: hepatocellular carcinoma", "Tumor", GSE214846_sample$sample_type)
GSE214846_sample$sample_type <- gsub("tissue: paracancerous normal tissues", "Normal", GSE214846_sample$sample_type)
GSE207435_sample$sample_type <- gsub("disease state: Tumor", "Tumor", GSE207435_sample$sample_type)
GSE207435_sample$sample_type <- gsub("disease state: Normal", "Normal", GSE207435_sample$sample_type)

mRNA_meta$sample_type <- gsub("Primary Tumor", "Tumor", mRNA_meta$sample_type)
mRNA_meta$sample_type <- gsub("Solid Tissue Normal", "Normal", mRNA_meta$sample_type)
mRNA_meta$sample_type <- as.factor(mRNA_meta$sample_type)
levels(mRNA_meta$sample_type)

mRNA_meta<- as.data.frame(mRNA_meta)
mRNA_meta<- mRNA_meta[, c(4,27)]
rownames(mRNA_meta)<- mRNA_meta[,1]
mRNA_meta$Batch <- "TCGA"
mRNA_meta<- mRNA_meta[,2:3]

mRNA_meta$sample_type <- gsub("Primary Tumor", "Tumor", mRNA_meta$sample_type)
mRNA_meta$sample_type <- gsub("Solid Tissue Normal", "Normal", mRNA_meta$sample_type)

merged_metadata <- rbind(GSE144269_sample, GSE214846_sample, GSE207435_sample, mRNA_meta)
merged_metadata$Sample<- rownames(merged_metadata)

colnames(merged_counts_raw) <- gsub("\\.", "-", colnames(merged_counts_raw))

merged_metadata <- merged_metadata[match(colnames(merged_counts_raw), merged_metadata$Sample), ]

write.csv(merged_metadata, "merged_metadata.csv") #Final file

library(edgeR)
library(sva)

dge <- DGEList(counts = merged_counts_raw)
raw_counts_for_combatseq <- merged_counts_raw[rownames(dge), ]

mod_bio <- model.matrix(~ sample_type, data = merged_metadata) # Corrected to sample_type
rownames(mod_bio) <- colnames(raw_counts_for_combatseq)

count_matrix <- as.matrix(merged_counts_raw)
adjusted_counts <- ComBat_seq(count_matrix, batch=merged_metadata$Batch, group=NULL, covar_mod=mod_bio)

cat("ComBat_seq correction complete. Dimensions of corrected data:", dim(combat_corrected_counts), "\n")


write.csv(adjusted_counts, "adjusted_counts.csv")
write.csv(merged_counts_raw, "merged_counts_raw.csv")
write.csv(merged_metadata, "merged_metadata.csv")

adjusted_counts<- data.frame(adjusted_counts)
adjusted_counts_2<- rbind(adjusted_counts, merged_metadata$sample_type)
row.names(adjusted_counts_2)[19074]<- "sample_type"
adjusted_counts_2<- t(adjusted_counts_2)
adjusted_counts_2<- as.data.frame(adjusted_counts_2)
adjusted_counts_2[, 1:19073] <- lapply(adjusted_counts_2[, 1:19073], as.numeric)
adjusted_counts_2$sample_type<- as.factor(adjusted_counts_2$sample_type)

adjusted_counts_2[, 1:19073] <- adjusted_counts_2[, 1:19073][rowSums(adjusted_counts_2[, 1:19073]) > 0, ] 

adjusted_counts_3<- adjusted_counts

colnames(adjusted_counts_3) <- gsub("\\.", "-", colnames(adjusted_counts_3))
merged_metadata$sample_type<- as.factor(merged_metadata$sample_type)

library(DESeq2)
RNA_counts <- DESeqDataSetFromMatrix(countData = adjusted_counts_3, colData = merged_metadata, design = ~sample_type)
merged_metadata$sample_type <- relevel(merged_metadata$sample_type, ref = "Normal")
levels(merged_metadata$sample_type)
RNA_counts <- DESeq(RNA_counts)
resultsNames(RNA_counts)
RNA_res <- results(RNA_counts, alpha = 0.01, contrast = c("sample_type", "Tumor", "Normal")) 

RNA_res  <- subset(RNA_res, padj < 0.5 & abs(log2FoldChange) > 2)
RNA_res  <- as.data.frame(RNA_res)
View(RNA_res) 
summary(RNA_res)
head(RNA_res)

write.csv(RNA_res, "DEGs.csv")