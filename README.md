# Multi-Omics Network Integration

> A tumour is not defined by its transcriptome alone. Or its methylome. Or its proteome. It's defined by all of them together. This project fuses multiple omics layers into a single patient similarity network to find clinically meaningful subtypes no single data type could reveal.

## The Limitation of Single-Omics Analysis

You run RNA-seq on 200 cancer patients. You cluster them into 3 subtypes. But are those real subtypes, or artefacts of looking through just one lens?

Methylation might reveal a subtype driven by epigenetic silencing that looks identical at the transcriptome level. Proteomics might show that two transcriptomically similar groups have completely different pathway activities. Each omics layer captures a different slice of biology.

## How Similarity Network Fusion Works

1. For each omics layer, build a patient-patient similarity network
2. Iteratively fuse the networks: if two patients are similar in both expression AND methylation, their connection strengthens
3. Cluster the fused network with spectral clustering
4. Validate subtypes against clinical outcomes

## Why This Matters for Precision Medicine

Multi-omics subtypes have been shown to better predict treatment response, survival outcomes, and disease progression than single-omics subtypes.

## Usage
```bash
python integrate_omics.py --expr data/expression.csv --methyl data/methylation.csv --protein data/proteomics.csv
```

## Validating Your Subtypes

Finding clusters is easy. Finding clinically meaningful clusters is hard. After running this, check survival differences between subtypes, look for enriched pathways per subtype, and test on an independent cohort if possible.
