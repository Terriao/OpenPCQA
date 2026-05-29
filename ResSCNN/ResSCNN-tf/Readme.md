# ResSCNN
key words: point cloud quality assessment, no-reference, deep learning, minkowski engine

ResSCNN is a no-reference quality cloud quality assessment(PCQA) model. This model applies the residual deep nueral network designed based on minkowski engine to extract the feature representation of point clouds, and it performs excellent on LS-PCQA-I dataset.


## Our Contributions
* We transplant from pytorch to tensorflow and mindspore using conventional layers to replace minkowski layers.
* Benchmark test on tensorflow and mindspore.

## File Structure
root


## Datasets
1. LS-PCQA

## requirements
1. tensorflow
* ubuntu 20.04 LTS
* cuda 11.6
* python 3.8.16
* tensorflow-gpu 2.11.0
* tensorflow-addons 0.19.0

# Performance
