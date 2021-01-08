# Simple Keras Convolutional Neural Network

A simple implementation of a CNN using Tensorflow / Keras and Cifar-10 Database

## Using Cifar 10 Dataset: [Cifar-10]('https://www.cs.toronto.edu/~kriz/cifar.html')

- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
- There are 50000 training images and 10000 test images.
- 32 x 32 = 1024 x 3 = ( 3072 input pixels )

### Image 32x32

![Cifar-Image](https://i.ibb.co/5cc6d24/cifar-image.png)

# Model

![Convolutional](https://i.ibb.co/y6W85Cw/convolutions.png)

| Type                       | Output Shape | Format                 |
|----------------------------|--------------|------------------------|
| Images                     | 32x32        |  input_type            |
| 2D Convolution(32 Filters) | 32x32x32     | (input_type)x(filters) |
| 2D Convolution(32 Filters) | 32x32x32     | (input_type)x(filters) |
| MaxPooling                 | 16x16x32     | (input_type)x(filters) |
| 2D Convolution(64 Filters) | 16x16x64     | (input_type)x(filters) |
| MaxPooling                 | 8x8x64       | (input_type)x(filters) |
| Flatten                    | 4096         |  input_type            |
| Input Dense NetWork        | 128          | neurons                |
| Output Dense Network       | 10 Classes   | neurons                |

### Classes

**['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']**

# Required tools

[python 3.8](https://www.python.org/download/releases/3.8/)

[tensorflow 2.4.0](https://pypi.org/project/tensorflow/2.4.0/)


