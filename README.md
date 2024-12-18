# Balanced Multi-Relational Graph Clustering

This repository contains the source code and datasets for the paper "Balanced Multi-Relational Graph Clustering", accepted by the 32nd ACM International Conference on Multimedia (MM 2024).

Paper Link: https://arxiv.org/abs/2407.16863

# Available Data

All the datasets and the trained model parameters can be downloaded from [datasets link](https://drive.google.com/file/d/18Nma11U2X4tvc_jvLYl1I3BpFbhRSR0A/view?usp=sharing).

Place the 'data' and 'best' folders from the downloaded files into the BMGC directory.

# Requirements

This code requires the following:

* Python==3.9.16
* PyTorch==1.13.1
* DGL==0.9.1
* Numpy==1.24.2
* Scipy==1.10.1
* Scikit-learn==1.2.1
* Munkres==1.1.4
* kmeans-pytorch==0.3
* PyTorch_Geometric==2.2.0

# Training

`python main.py` or `python large.py` (to run the MAG dataset)

# MAG Dataset

MAG is a large-scale citation network, constituting the largest dataset in multi-relation graph clustering thus far. MAG is a subset we extracted from [OGBN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag), consisting of the four largest classes. MAG contains 113,919 papers with graphs generated by two meta-paths (paper-author-paper and paper-paper). Each paper is associated with a 128-dimensional word2vec feature vector. The processed MAG can be downloaded from the above "datasets link".

If you use the MAG dataset in your research, please make sure to cite our paper. Thank you!

# BibTeX

```
@inproceedings{shen2024balanced,
  title={Balanced Multi-Relational Graph Clustering},
  author={Shen, Zhixiang and He, Haolan and Kang, Zhao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={4120--4128},
  year={2024}
}
```
