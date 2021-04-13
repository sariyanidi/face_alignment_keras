This is the code for training a face alignment method with keras. Here we publish only code for training. For inference (i.e., testing) we suggest you to use our OpenCV-based inference code in https://github.com/sariyanidi/face_alignment_opencv, which is a minimal-dependence repo for inference. 

Here we re-implement a state-of-the-art face alinment method, namely 2D-FAN, on keras. There are two main reasons that we re-implement this method: 
- The original code's training module is implemented with Lua Torch, which is difficult to run with recent versions of CUDA etc. Moreover, the inference code for the original method is implemented with PyTorch, and the conversion from a Lua Torch model to a PyTorch model is not trivial.
- We propose a minimal-dependence implementation for inference that uses nothing but OpenCV. In this repository we show how to train a model on keras and how to convert it into a model readable by OpenCV. Our inference method shows

The instructions to obtain the training data are provided in data/README.md
