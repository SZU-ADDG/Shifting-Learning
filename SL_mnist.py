import os
import math
import time
import torch

import Dataset as Ds
import SL_Layer as mySNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # gpu


def MNIST_train(sn, bs, data_set="mnist"):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("Training on cpu.")
        device = torch.device("cpu")
    else:
        print("Training on gpu.")
        device = torch.device("cuda:0")

    # Network layout
    ly1 = mySNN.SNNLayer(inCh=784, outCh=800)
    ly2 = mySNN.SNNLayer(inCh=800, outCh=10)

    # send parameters to device
    ly1.e_ts, ly1.e_tp = ly1.e_ts.to(device), ly1.e_tp.to(device)
    ly2.e_ts, ly2.e_tp = ly2.e_ts.to(device), ly2.e_tp.to(device)
    ly1.cause_mask = ly1.cause_mask.to(device)
    ly2.cause_mask = ly2.cause_mask.to(device)
    ly1.adam_m_Ets, ly1.adam_m_Etp = ly1.adam_m_Ets.to(device), ly1.adam_m_Etp.to(device)
    ly2.adam_m_Ets, ly2.adam_m_Etp = ly2.adam_m_Ets.to(device), ly2.adam_m_Etp.to(device)
    ly1.adam_v_Ets, ly1.adam_v_Etp = ly1.adam_v_Ets.to(device), ly1.adam_v_Etp.to(device)
    ly2.adam_v_Ets, ly2.adam_v_Etp = ly2.adam_v_Ets.to(device), ly2.adam_v_Etp.to(device)

    # datasets
    data_loader = Ds.mnist_train_loader
    if data_set == "fashion_mnist":
        data_loader = Ds.fashion_mnist_train_loader

    # Data prepare
    X_train, y_train = [], []  # (60000,1,28,28)
    for idx, (data, target) in enumerate(data_loader):  # read out all data in one time
        X_train, y_train = data, target
    # X_train = torch.where(X_train > 0.5, 0.01, 2.3)
    # X_train = 2.9 * (1.0 - X_train)
    # X_train = torch.where(X_train >= 2.9, 2.9, X_train)
    # X_train = 8.0 * X_train
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Training process
    epoch_num = 20
    lr_start, lr_end = 1e-4, 1e-6  # decaying learning rate for shifting timings
    lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)
    lr_Etp = 1e-4

    bn = int(math.ceil(sn / bs))
    loss, total_loss = 0, []
    acc, total_acc = 0, []

    time_start = time.time()  # time when training process start
    for epoch in range(epoch_num):  # 6000
        lr_Ets = lr_start * lr_decay ** epoch
        bs0 = bs
        for bi in range(bn):  # 20
            # input data
            if (bi + 1) * bs0 > sn:
                data, tar = X_train[bi * bs0:sn], y_train[bi * bs0:sn]
            else:
                data, tar = X_train[bi * bs0:(bi + 1) * bs0], y_train[bi * bs0:(bi + 1) * bs0]
            z0 = torch.exp(1.0 - data.view(-1, 28 * 28))  # processing data (bs,1,28,28) --> (bs,784)
            tar_10 = (torch.ones(tar.size()[0], 10)*0.99).to(device)  # the prepared label
            for i in range(data.size()[0]):
                tar_10[i, tar[i]] = 0.01

            bs0 = z0.size()[0]

            # Forward propagation
            z1 = ly1.forward(bs0, z0, dv=device)
            z2 = ly2.forward(bs0, z1, dv=device)

            # Shifting Learning
            z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(tar_10), dim=1)
            delta2 = z2_lo - z_tar
            delta1 = ly2.pass_delta(bs0, delta2)

            ly2.backward(bs0, delta2, z1, z2, lr_Ets, lr_Etp)
            ly1.backward(bs0, delta1, z0, z1, lr_Ets, lr_Etp)

            loss = torch.sum((z2_lo - z_tar) * (z2_lo - z_tar)) / data.size()[0]
            if bi % 100 == 0:
                print("Current Training epoch: " + str(epoch + 1), end="\t")
                print("Progress: [" + str(bi * bs0) + "/" + str(sn), end="")
                print("(%.0f %%)]" % (100.0 * bi * bs0 / sn), end="\t")
                print("Error: " + str(loss))
                time_cur = time.time()
                print("Time consuming: %.3f s" % (time_cur - time_start))
        pass  # bi
        total_loss.append(loss)
        torch.save(ly1.e_ts, "./parameters_record/SL_mnist_ets1")
        torch.save(ly1.e_tp, "./parameters_record/SL_mnist_etp1")
        torch.save(ly2.e_ts, "./parameters_record/SL_mnist_ets2")
        torch.save(ly2.e_tp, "./parameters_record/SL_mnist_etp2")
        # 每轮测试一次
        correct_temp, sn_temp = MNIST_test(10000, 100, "mnist")
        acc = correct_temp / sn_temp
        print("Accuracy: " + str(int(correct_temp)) + "/" + str(sn_temp), end="")
        print("(%.3f %%)" % (100. * acc))
        total_acc.append(acc)
    pass  # epoch
    time_end = time.time()  # time when training process end
    print("Time consuming: %.3f s" % (time_end - time_start))
    torch.save(ly1.e_ts, "./parameters_record/SL_mnist_ets1")
    torch.save(ly1.e_tp, "./parameters_record/SL_mnist_etp1")
    torch.save(ly2.e_ts, "./parameters_record/SL_mnist_ets2")
    torch.save(ly2.e_tp, "./parameters_record/SL_mnist_etp2")
    total_loss = torch.tensor(total_loss)
    torch.save(total_loss, "./results_record/SL_MNIST_loss")
    torch.save(total_acc, "./results_record/SL_MNIST_acc")


def MNIST_test(sn, bs, data_set="mnist", isConMx=False):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        # print("Testing on cpu.")
        device = torch.device("cpu")
    else:
        # print("Testing on gpu.")
        device = torch.device("cuda:0")

    e_ts1 = torch.load("./parameters_record/SL_mnist_ets1", map_location=torch.device('cpu')).to(device)
    e_tp1 = torch.load("./parameters_record/SL_mnist_etp1", map_location=torch.device('cpu')).to(device)
    e_ts2 = torch.load("./parameters_record/SL_mnist_ets2", map_location=torch.device('cpu')).to(device)
    e_tp2 = torch.load("./parameters_record/SL_mnist_etp2", map_location=torch.device('cpu')).to(device)

    ly1 = mySNN.SNNLayer(784, 800, e_ts=e_ts1, e_tp=e_tp1)
    ly2 = mySNN.SNNLayer(800, 10, e_ts=e_ts2, e_tp=e_tp2)

    # datasets
    data_loader = Ds.mnist_test_loader
    if data_set == "fashion_mnist":
        data_loader = Ds.fashion_mnist_test_loader

    # Data prepare
    X_test, y_test = [], []
    for idx, (data, target) in enumerate(data_loader):
        X_test, y_test = data, target
    X_test = torch.where(X_test > 0.5, 0.01, 2.3)
    # X_test = 2.9 * (1.0 - X_test)
    # X_test = torch.where(X_test >= 2.9, math.inf, X_test)
    # X_test = 8.0 * X_test
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Testing Process
    correct = 0
    conMx = torch.zeros((10, 10))
    bn = int(math.ceil(sn / bs))
    for bi in range(bn):
        if (bi + 1) * bs > sn:
            data, tar = X_test[bi * bs:sn], y_test[bi * bs:sn]
        else:
            data, tar = X_test[bi * bs:(bi + 1) * bs], y_test[bi * bs:(bi + 1) * bs]
        z0 = torch.exp(1.0 - data.view(-1, 28 * 28))

        # Forward propagation
        z1 = ly1.forward(bs, z0, dv=device)
        z2 = ly2.forward(bs, z1, dv=device)

        lo = torch.softmax(z2, dim=1)
        prediction = torch.argmax(lo, dim=1)
        correct += prediction.eq(tar.data).sum()

        if isConMx:
            for k in range(bs):
                conMx[tar.data[k], prediction[k]] += 1
        #print(bi)
        #print(prediction.size())
        #print(prediction)
        #print(tar.data.size())
        #print(tar.data)
    pass
    if isConMx:
        print(conMx)
        torch.save(conMx, "./results_record/SL_MNIST_conMx")
    return correct, sn


def run_MNIST():
    # MNIST_train(60000, 128, "mnist")
    correct, sn = MNIST_test(10000, 100, "mnist")

    print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
    print("(%.3f %%)" % (100. * correct / sn))


def main():
    run_MNIST()


if __name__ == "__main__":
    main()
