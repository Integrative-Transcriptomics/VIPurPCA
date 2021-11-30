#!/usr/bin/python
# from mnist import MNIST
# import matplotlib.pyplot as plt
# import numpy as np
# from itertools import compress
#
# def get_mnist_dataset(digit, n_samples):
#     mndata = MNIST('/share/home/zabel/MXNet/data/MNIST')
#     images, labels = mndata.load_training()
#     n_labels = 10
#     image_size = 28
#     image_pixels = image_size * image_size
#
#     # get only specified digit
#     fives_b = [True if labels[i] == digit else False for i in range(len(labels))]
#     fives_img = list(compress(images, fives_b))[0:n_samples]
#     fives_lab = list(compress(labels, fives_b))[0:n_samples]
#     return image_size, image_pixels, fives_lab, fives_img
