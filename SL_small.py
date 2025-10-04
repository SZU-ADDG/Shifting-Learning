import math
import time
import torch

import Dataset as Ds
import SL_Layer as mySNN


def Train(X, y, f=0, siz=0, st=None, totSn=0, sn=0, bs=0, epN=0, inv=0, lr=None, name=None):
    # st = [inCh, hiCh, outCh], lr = [lr_start, lr_end, lr_Etp], name = string
    # Iris default: (X, y, f=0, siz=30,
    # st=[4, 10, 3], totSn=150, sn=120, bs=30, epN=400, inv=50, lr=[1e-2, 1e-2, 1e-6], name="Iris")

    # Network layout
    ly1 = mySNN.SNNLayer(inCh=st[0], outCh=st[1])  # Network structure
    ly2 = mySNN.SNNLayer(inCh=st[1], outCh=st[2])

    # Read out data, totSn: total sample num
    X_train, y_train = torch.cat(
        (X[0:f * siz], X[f*siz + siz:totSn]), dim=0), torch.cat((y[0:f*siz], y[f*siz + siz:totSn]), dim=0)

    # Training process
    epoch_num, interval = epN, inv
    lr_start, lr_end = lr[0], lr[1]  # decaying learning rate for shifting timings
    lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)
    lr_Etp = lr[2]

    bn = int(math.ceil(sn / bs))  # number of batches
    loss, total_loss = 0, []
    for epoch in range(epoch_num):
        lr_Ets = lr_start * lr_decay ** epoch
        for bi in range(bn):
            # input data
            if (bi + 1) * bs > sn:  # for the last batch with unusual size
                data, tar = X_train[bi*bs:sn], y_train[bi*bs:sn]
            else:  # for other batches
                data, tar = X_train[bi*bs:(bi + 1)*bs], y_train[bi*bs:(bi + 1)*bs]
            z0 = torch.exp(1.0 - data)  # (bs=30, 4) temporal-coding

            # Forward propagation
            z1 = ly1.forward(bs, z0)
            z2 = ly2.forward(bs, z1)

            # Shifting Learning
            z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(1.0 - tar), dim=1)
            delta2 = z2_lo - z_tar
            delta1 = ly2.pass_delta(bs, delta2)

            ly2.backward(bs, delta2, z1, z2, lr_Ets, lr_Etp)
            ly1.backward(bs, delta1, z0, z1, lr_Ets, lr_Etp)

            loss = torch.sum((z2_lo - z_tar) * (z2_lo - z_tar)) / data.size()[0]
        total_loss.append(loss)
        if epoch % interval == 0:
            print("Current Training epoch: " + str(epoch + 1), end="\t")
            print("Progress: [" + str(epoch) + "/" + str(epoch_num), end="")
            print("(%.0f %%)]" % (100.0 * epoch / epoch_num), end="\t")
            print(" ")
            print("Error: " + str(loss))
        torch.save(ly1.e_ts, "./parameters_record/" + name + "/SL_" + name + "_e_ts1")
        torch.save(ly1.e_tp, "./parameters_record/" + name + "/SL_" + name + "_e_tp1")
        torch.save(ly2.e_ts, "./parameters_record/" + name + "/SL_" + name + "_e_ts2")
        torch.save(ly2.e_tp, "./parameters_record/" + name + "/SL_" + name + "_e_tp2")
    total_loss = torch.tensor(total_loss)
    torch.save(total_loss, "./results_record/SL_" + name + "_loss")
    # print(total_loss)


def Test(X, y, st=None, sn=0, bs=0, name=None):
    # Iris default: (X_test, y_test, st=[4, 10, 3], sn=30, bs=30, name="Iris")

    # Network layout
    e_ts1 = torch.load("./parameters_record/" + name + "/SL_" + name + "_e_ts1")
    e_tp1 = torch.load("./parameters_record/" + name + "/SL_" + name + "_e_tp1")
    e_ts2 = torch.load("./parameters_record/" + name + "/SL_" + name + "_e_ts2")
    e_tp2 = torch.load("./parameters_record/" + name + "/SL_" + name + "_e_tp2")

    # Network layout
    ly1 = mySNN.SNNLayer(inCh=st[0], outCh=st[1], e_ts=e_ts1, e_tp=e_tp1)
    ly2 = mySNN.SNNLayer(inCh=st[1], outCh=st[2], e_ts=e_ts2, e_tp=e_tp2)

    # Testing Process
    correct = 0
    pred_tot, tar_tot = [], []
    bn = int(math.ceil(sn / bs))  # number of batches
    for bi in range(bn):
        if (bi + 1) * bs > sn:
            data, tar = X[bi * bs:sn], y[bi * bs:sn]
        else:
            data, tar = X[bi * bs:(bi + 1) * bs], y[bi * bs:(bi + 1) * bs]
        z0 = torch.exp(1.0 - data)  # (bs=30, 4)
        tar = torch.argmax(tar, dim=1)

        # Forward propagation
        z1 = ly1.forward(bs, z0)
        z2 = ly2.forward(bs, z1)

        prediction = torch.argmin(z2, dim=1)
        prob = torch.tensor(1.0 - torch.softmax(torch.log(z2), dim=1)[:, 1])
        correct += prediction.eq(tar.data).sum()

        pred_tot.append(prob)
        tar_tot.append(tar)
    pass
    torch.save(pred_tot, "./results_record/SL_" + name + "_pred")
    torch.save(tar_tot, "./results_record/SL_" + name + "_tar")
    return correct, sn


def Run(loader=None, f=0, siz=0, st=None, totSn=0, sn=0, bs=0, epN=0, inv=0, lr=None, name=None):
    # Data prepare
    X, y = loader
    X_test, y_test = X[f*siz:f*siz + siz], y[f*siz:f*siz + siz]

    time_start = time.time()
    Train(X, y, f=f, siz=siz, st=st, totSn=totSn, sn=sn, bs=bs, epN=epN, inv=inv, lr=lr, name=name)
    correct, sn = Test(X_test, y_test, st=st, sn=bs, bs=bs, name=name)
    time_end = time.time()

    print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
    print("(%.3f %%)" % (100. * correct / sn))
    print("Time consuming: %.3f s" % (time_end - time_start))


def run_Task(siz=0, name=None):
    if name == "Iris":
        Run(loader=Ds.iris_loader(), f=0, siz=siz,
            st=[4, 10, 3], totSn=150, sn=120, bs=30, epN=400, inv=50, lr=[1e-2, 1e-2, 1e-6], name="Iris")


def Iris_fold():
    # Data prepare
    X, y = Ds.iris_loader()  # X for inputs, y for label
    siz = 30

    time_start = time.time()
    total_correct = 0
    for f in range(5):  # 5-fold cross validation
        X_test, y_test = X[f * siz:f * siz + siz], y[f * siz:f * siz + siz]

        print("Fold: " + str(f+1) + ".")
        Train(X, y, f=f, siz=siz,
              st=[4, 10, 3], totSn=150, sn=120, bs=30, epN=400, inv=50, lr=[1e-2, 1e-2, 1e-6], name="Iris")
        correct, sn = Test(X_test, y_test, st=[4, 10, 3], sn=30, bs=30, name="Iris")
        total_correct += int(correct)
        print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
        print("(%.3f %%)" % (100. * correct / sn))
        print(" ")
    pass
    time_end = time.time()

    print(" ")
    print("5-fold accuracy: " + str(total_correct/5) + "/" + str(siz), end="")
    print("(%.3f %%)" % (100. * (total_correct/5) / siz))
    print("Time consuming: %.3f s" % (time_end - time_start))


def main():
    run_Task(siz=30, name="Iris")
    # Iris_fold()


if __name__ == "__main__":
    main()
