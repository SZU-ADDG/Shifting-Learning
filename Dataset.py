import numpy
import pandas

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from skimage import feature


# iris dataset
def iris_loader():
    row_data = numpy.asmatrix(pandas.read_csv("D:/Dataset/Data/iris/iris.data", header=None, sep=','))  # read file for data
    row_data1, row_data2, row_data3 = row_data[0:50, :], row_data[50:100, :], row_data[100:150, :]
    row_data1[:, 4], row_data2[:, 4], row_data3[:, 4] = 0, 1, 2  # Setosa-0, Versicolour-1, Virginica-2
    row_data = numpy.row_stack((row_data1, row_data2, row_data3))

    tensor_data = torch.from_numpy(row_data.astype(float))  # disrupt the order
    rand_index = torch.tile(torch.reshape(torch.randperm(150), [150, 1]), [1, 5])
    tensor_data = torch.gather(tensor_data, dim=0, index=rand_index)

    tensor_tars = torch.ones((150, 3)) * 0.01  # label one-hot coding
    for i in range(150):
        tensor_tars[i, int(tensor_data[i, 4])] = 0.99

    tensor_inputs = tensor_data[:, 0:4]  # input normalize
    data_max, data_min = torch.max(tensor_inputs), torch.min(tensor_inputs)
    tensor_inputs = (tensor_inputs - data_min)/(data_max - data_min) + 0.01
    return tensor_inputs, tensor_tars


# MNIST
mnist_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("D:/Dataset/mnist", train=True, download=True,  # in MNIST/raw
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=60000, shuffle=True
)
mnist_test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("D:/Dataset/mnist", train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=10000, shuffle=False
)

# FashionMNIST
fashion_mnist_train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("D:/Dataset/mnist", train=True, download=True,  # in FashionMNIST/raw
                          transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=60000, shuffle=True
)
fashion_mnist_test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("D:/Dataset/mnist", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=10000, shuffle=False
)


def mnist_train_border_detection():
    X_train, y_train = [], []
    for idx, (data, target) in enumerate(mnist_train_loader):
        X_train, y_train = data, target
    for item in X_train:
        item[0] = torch.tensor(feature.canny(numpy.array(item[0]), sigma=0.5))
    return X_train, y_train


def mnist_test_border_detection():
    X_test, y_test = [], []
    for idx, (data, target) in enumerate(mnist_test_loader):
        X_test, y_test = data, target
    for item in X_test:
        item[0] = torch.tensor(feature.canny(numpy.array(item[0]), sigma=0.5))
    return X_test, y_test


# Show images in MNIST & FashionMNIST dataset
def MNIST_show(data_set="mnist", show_part="train"):
    train_data_loader = mnist_train_loader
    if data_set == "fashion_mnist":
        train_data_loader = fashion_mnist_train_loader
    test_data_loader = mnist_test_loader
    if data_set == "fashion_mnist":
        test_data_loader = fashion_mnist_test_loader

    X_train, y_train = [], []
    for idx, (data, target) in enumerate(train_data_loader):  # read out all data in a time
        X_train, y_train = data, target
    print("Form of train set data: " + str(X_train.shape))  # (60000,1,28,28)
    print("Form of train set label: " + str(y_train.shape))  # (60000)
    print("")

    X_test, y_test = [], []
    for idx, (data, target) in enumerate(test_data_loader):
        X_test, y_test = data, target
    print("Form of test set data: " + str(X_test.shape))  # (10000,1,28,28)
    print("Form of test set label: " + str(y_test.shape))  # (10000)
    print("")

    if show_part == "train":
        for i in range(X_train.size()[0]):   # show train set
            data, target = X_train[i], y_train[i]
            print("Image" + str(i+1) + "label: " + str(target))
            # img = data.permute(1, 2, 0)
            x_img = data[0]
            x_img = torch.where(x_img > 0.5, 0.01, 1.79)
            # x_img = numpy.array(data[0])
            # x_img = feature.canny(x_img, sigma=0.5)  # border detection
            print(x_img)
            plt.imshow(x_img)
            plt.show()
    else:
        for i in range(X_test.size()[0]):  # show test set
            data, target = X_test[i], y_test[i]
            print("Image" + str(i+1) + "label: " + str(target))
            img = data.permute(1, 2, 0)
            plt.imshow(img)
            plt.show()


def main():
    MNIST_show(data_set="fashion_mnist", show_part="train")


if __name__ == "__main__":
    main()
