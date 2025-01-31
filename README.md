# A Flag Decomposition for Hierarchical Data Sets



![_](concept.png)

## Abstract
Flag manifolds encode hierarchical nested sequences of subspaces and serve as powerful structures for various computer vision and machine learning applications. Despite their utility in tasks such as dimensionality reduction, motion averaging, and subspace clustering, current applications are often restricted to extracting flags using common matrix decomposition methods like the singular value decomposition. Here, we address the need for a general algorithm to factorize and work with hierarchical datasets. In particular, we propose a novel, flag-based method that decomposes arbitrary hierarchical real-valued data into a hierarchy-preserving flag representation in Stiefel coordinates. Our work harnesses the potential of flag manifolds in applications including denoising, clustering, and few-shot learning.


## Quick Start
1. Install [anaconda](https://www.anaconda.com/download) or [miniconda](https://docs.anaconda.com/free/miniconda/index.html)

1. Initialize conda environment
    ```
    conda env create -f flag_decomp.yml
    conda activate flag_decomp
    ```

1. Visit getting_started.ipynb for an introduction to flag decompoisitions for flag recovery and denoising



## Authors

Nathan Mankovich, Ignacio Santamaria, Gustau Camps-Valls, and Tolga Birdal

