# LeNet-1 via PyTorch

### Intro
LeNet-1 is one of the first Convolutional Neural Network (CNN) models proposed for image classification.
The main purpose of this neural network is recognize and classify written digits (0-9).

### Links
- [Original Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf)
- [Helpful article on Medium](https://medium.com/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17)

### Disclaimer

I wrote this code solely for educational purposes. Prior to implementing it, I have learned different regularization techniques, backpropagation, layer types, optimization modes etc., but all of them were either too theoretical or implemented via some custom framework. 

After undersatnding how to inplement all these basics from scratch, I wanted to go through the whole process of:
- Reading a white paper
- Writing a Python code on a recognized framework (i.e. on PyTorch)
- Training the model (analyzing how to deal with overfitting, hyperparameter tuning etc.)

LeNet-1 seems to be a perfect choice for that because:
- The paper is easy to read and implement
- Architecture is trivial:
![Image of the architecture](https://github.com/grvk/lenet-1/blob/master/data/LeNet-1-architecture.png?raw=true)
- LeNet-1 is even simpler than LeNet-5 (i.e. it has fewer fully connected layers), and thus faster to train
- It can be trained on CPU (takes about 1-4 hours depending on the machine)

# This implementation exactly follows the one described in the original paper
It means that it doesn't use ReLU, normalization techniques, or any optimization algorithms that were proposed later. It uses a classic Stochastic Gradient Descent (without a momentum). It made the implementaton a bit slower/dumber than it could've been, but also (hopefully) easier to understand.

**This implementation also contains helper code and training output obtained during hyperparameter tuning.** This shows the overall thinking process rather than just a ready solution. If you are interested to see just the results, scroll to the bottom of the notebook.

Last disclaimer: **By no means it's the cleanest/best solution. The code could be refactored (slightly restructured, with extra fail-checks, enhanced documentation), but this was not the purpose of this endeavor. The main point was to go through actually get started with:**
- PyTorch, Python 3, NumPy, MatPlotLib, jupyter, CNN model training and imnplementation

# Installation

```sh
$ conda env create -f environment.yml
$ conda activate LeNet-1
$ jupyter notebook
```

# Results
- Accuracy on the test dataset: 98.62%

Train loss vs Validation loss chart
![Chart image](https://github.com/grvk/lenet-1/blob/master/data/optim_model_loss_chart.png?raw=true)

### License
MIT


