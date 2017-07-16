# FFTLasso: Large-Scale LASSO in the Fourier Domain

Authors: Adel Bibi, Hani Itani and Bernard Ghanem

Project Website: https://ivul.kaust.edu.sa/Pages/pub-fft-lasso.aspx

Personal Website: www.adelbibi.com

License: See LICENSE file

**If you use any of this work please cite:**  
Adel Bibi, Hani Itani, Bernard Ghanem  
"FFTLasso: Large-Scale LASSO in the Fourier Domain"  
Conference on Computer Vision and Pattern Recognition (CVPR 2017) [Oral]

The code is tested on windows. The required depdencies are:

a) tensorflow-gpu 1.0.1

b) Jupter notebook

c) scipy 0.19.0

d) numpy 1.12.1


The subdirectory "FFTLasso_TensorFlow_VerticalSplits" demonstrates how FFTLasso can be implemented with arbitrary number of vertical splits of the dictionary A to be distrubted over multiple GPUs.

Note: Tensorflow in general is much slower than having a CUDA version. This is a simple demo of how the proposed approach can trivially parallelize data over multiple GPUs since the operations involved are elementwise and FFTs. 
