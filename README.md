# Multi-Omics Network Integration

Integrating transcriptomics, proteomics, and methylation data using Similarity Network Fusion and Graph Neural Networks for patient stratification.

## Features
- Similarity Network Fusion (SNF) for multi-omics integration
- Patient similarity networks from each omics layer
- Spectral clustering on fused network
- GNN-based classification on patient graphs
- Survival analysis per cluster
- Network visualisation

## Methods
1. Construct patient similarity networks per omics layer
2. Fuse networks using SNF iterative approach
3. Spectral clustering for patient subtyping
4. Validate clusters with survival data

## Usage
```bash
python integrate_omics.py --expr data/expression.csv --methyl data/methylation.csv --protein data/proteomics.csv
```
