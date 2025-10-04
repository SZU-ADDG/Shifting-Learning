import time
import numpy
import torch
import scipy.io as scio

import SL_Layer as normal_Ly
import SL_MultiInputLy as input_Ly


def preprocess():
    # load the dataset spike
    train_data_raw = scio.loadmat("D://Dataset//spike//gamma_5-inhomogbgnoise-trials-250-syn-500-feat-9-train")
    train_data = train_data_raw['data']  # 200 trials, 8 colume, 500 cells in (1,500), cell structure(1,length about 10)
    test_data_raw = scio.loadmat("D://Dataset//spike//gamma_5-inhomogbgnoise-trials-250-syn-500-feat-9-validation")
    test_data = test_data_raw['data']  # 50 trials, same structure with above

    train_trials = 200
    test_trials = 50
    N_syn = 500  # number of synapses

    train_targets = numpy.zeros(train_trials)  # gain the training targets (sum of all the rewards)
    for i in range(train_trials):
        train_targets[i] = numpy.sum(train_data[i, 4])
    train_targets = torch.tensor(train_targets).to(torch.int)  # (200) max-21

    test_targets = numpy.zeros(test_trials)  # test targets
    for i in range(test_trials):
        test_targets[i] = numpy.sum(test_data[i, 4])
    test_targets = torch.tensor(test_targets).to(torch.int)  # (50) max-19

    train_samples = train_data[:, 0]  # (200-500-)
    train_spike, train_idx = [], []  # containing 200 trials
    for i in range(train_trials):  # train_trials
        current_sample = train_samples[i][0]
        spike_seq, idx_seq = torch.zeros(0), torch.zeros(0)
        for j in range(N_syn):  # N_syn
            spike_seq_j = current_sample[j][0]
            spike_seq_j = torch.tensor(spike_seq_j)  # spike sequence received by a single synapse in a trial
            idx_seq_j = torch.ones(spike_seq_j.size()[0]) * j  # synapse index for the spike
            spike_seq = torch.cat([spike_seq, spike_seq_j], dim=0)
            idx_seq = torch.cat([idx_seq, idx_seq_j], dim=0)
        pass
        spike_sorted, sort_idx = torch.sort(spike_seq, dim=0)
        idx_sorted = torch.gather(idx_seq, dim=0, index=sort_idx)
        print(spike_sorted.size())
        train_spike.append(spike_sorted)
        train_idx.append(idx_sorted.to(torch.int64))
    pass
    torch.save(train_spike, "./preprocessed_data/train_spike")
    torch.save(train_idx, "./preprocessed_data/train_idx")
    torch.save(train_targets, "./preprocessed_data/train_targets")

    test_samples = test_data[:, 0]  # (50-500-)
    test_spike, test_idx = [], []  # containing 50 trials
    for i in range(test_trials):  # test_trials
        current_sample2 = test_samples[i][0]
        spike_seq2, idx_seq2 = torch.zeros(0), torch.zeros(0)
        for j in range(N_syn):  # N_syn
            spike_seq_j2 = current_sample2[j][0]
            spike_seq_j2 = torch.tensor(spike_seq_j2)  # spike sequence received by a single synapse in a trial
            idx_seq_j2 = torch.ones(spike_seq_j2.size()[0]) * j  # synapse index for the spike
            spike_seq2 = torch.cat([spike_seq2, spike_seq_j2], dim=0)
            idx_seq2 = torch.cat([idx_seq2, idx_seq_j2], dim=0)
        pass
        spike_sorted2, sort_idx2 = torch.sort(spike_seq2, dim=0)
        idx_sorted2 = torch.gather(idx_seq2, dim=0, index=sort_idx2)
        print(spike_sorted2.size())
        test_spike.append(spike_sorted2)
        test_idx.append(idx_sorted2.to(torch.int64))
    pass
    torch.save(test_spike, "./preprocessed_data/test_spike")
    torch.save(test_idx, "./preprocessed_data/test_idx")
    torch.save(test_targets, "./preprocessed_data/test_targets")


def Spike_train():
    train_spk = torch.load("./preprocessed_data/train_spike")
    train_idx = torch.load("./preprocessed_data/train_idx")
    train_targets = torch.load("./preprocessed_data/train_targets")

    test_spk = torch.load("./preprocessed_data/test_spike")
    test_idx = torch.load("./preprocessed_data/test_idx")
    test_targets = torch.load("./preprocessed_data/test_targets")

    ly1 = input_Ly.SNN_MInLy(inCh=500, outCh=3200)
    ly2 = normal_Ly.SNNLayer(inCh=3200, outCh=22)

    # Training process
    epoch_num = 200
    lr_start, lr_end = 1e-5, 1e-6  # decaying learning rate for shifting timings
    lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)
    lr_Etp = 1e-6

    lr_Ets2 = 1e-5
    lr_Etp2 = 0.0
    sn = 200
    loss, total_loss = 0, []
    acc, total_acc = 0, []

    time_start = time.time()
    for epoch in range(epoch_num):  # epoch_num
        lr_Ets = lr_start * lr_decay ** epoch
        for bi in range(sn):  # sn
            bs = 1
            tar_22 = (torch.ones(bs, 22) * 0.99)  # 0-21
            tar_22[0, train_targets[bi]] = 0.01  # one-hot encoding

            # Forward propagation
            z1 = ly1.forward(train_spk[bi], train_idx[bi])
            z2 = ly2.forward(bs, z1)

            # Shifting Learning
            z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(tar_22), dim=1)
            w2Ex = torch.tile(torch.reshape(ly2.e_ts, [1, ly2.inCh, ly2.outCh]), [bs, 1, 1])
            delta2 = (z2_lo - z_tar) / (torch.sum(w2Ex * ly2.cause_mask, dim=1) - ly2.th)

            inNum, outCh = train_spk[bi].size()[0], 3200
            idxEx = torch.tile(torch.reshape(train_idx[bi], [inNum, 1]), [1, outCh])
            w1Ex = torch.gather(ly1.e_ts, dim=0, index=idxEx)
            #delta2 = z2_lo - z_tar
            #print(ly1.cause_mask.size())
            #print(w1Ex.size())
            delta1 = ly2.pass_delta(bs, delta2) / (torch.sum(w1Ex * ly1.cause_mask, dim=0) - ly1.th)

            ly2.backward(bs, delta2, z1, z2, lr_Ets, lr_Etp)
            ly1.backward(delta1, train_spk[bi], train_idx[bi], z1, lr_Ets, lr_Etp)

            loss = torch.sum((z2_lo - z_tar) * (z2_lo - z_tar))
            if bi % 50 == 0:
                print("Current Training epoch: " + str(epoch + 1), end="\t")
                print("Progress: [" + str(bi * bs) + "/" + str(sn), end="")
                print("(%.0f %%)]" % (100.0 * bi * bs / sn), end="\t")
                print("Error: " + str(loss*1000.0))
                #print(z2_lo)
                #print(z_tar)
                time_cur = time.time()
                print("Time consuming: %.3f s" % (time_cur - time_start))
        pass  # bi
        total_loss.append(loss)
        torch.save(ly1.e_ts, "./parameters_record/TCA/SL_spike_ets1")
        torch.save(ly1.e_tp, "./parameters_record/TCA/SL_spike_etp1")
        torch.save(ly2.e_ts, "./parameters_record/TCA/SL_spike_ets2")
        torch.save(ly2.e_tp, "./parameters_record/TCA/SL_spike_etp2")
        # test one time in a round
        correct_tr, sn_tr = Spike_test(train_spk, train_idx, train_targets)
        acc_tr = correct_tr / sn_tr
        print("Accuracy: " + str(int(correct_tr)) + "/" + str(sn_tr), end="")
        print("(%.3f %%)" % (100. * acc_tr))
        total_acc.append(acc_tr)

        '''correct_ts, sn_ts = Spike_test(test_spk, test_idx, test_targets)
        acc_ts = correct_ts / sn_ts
        print("Test Accuracy: " + str(int(correct_ts)) + "/" + str(sn_ts), end="")
        print("(%.3f %%)" % (100. * acc_ts))'''
    pass  # epoch
    time_end = time.time()  # time when training process end
    print("Time consuming: %.3f s" % (time_end - time_start))
    torch.save(ly1.e_ts, "./parameters_record/TCA/SL_spike_ets1")
    torch.save(ly1.e_tp, "./parameters_record/TCA/SL_spike_etp1")
    torch.save(ly2.e_ts, "./parameters_record/TCA/SL_spike_ets2")
    torch.save(ly2.e_tp, "./parameters_record/TCA/SL_spike_etp2")
    total_loss = torch.tensor(total_loss)
    torch.save(total_loss, "./results_record/SL_spike_loss")
    torch.save(total_acc, "./results_record/SL_spike_acc")


def Spike_test(spk, idx, targets):
    e_ts1 = torch.load("./parameters_record/TCA/SL_spike_ets1")
    e_tp1 = torch.load("./parameters_record/TCA/SL_spike_etp1")
    e_ts2 = torch.load("./parameters_record/TCA/SL_spike_ets2")
    e_tp2 = torch.load("./parameters_record/TCA/SL_spike_etp2")

    ly1 = input_Ly.SNN_MInLy(inCh=500, outCh=3200, e_ts=e_ts1, e_tp=e_tp1)
    ly2 = normal_Ly.SNNLayer(inCh=3200, outCh=22, e_ts=e_ts2, e_tp=e_tp2)

    sn = targets.size()[0]
    correct = 0
    for bi in range(sn):  # sn
        bs = 1
        tar_22 = (torch.ones(1, 22) * 0.99)  # 0-21
        tar_22[0, targets[bi]] = 0.01  # one-hot encoding

        # Forward propagation
        z1 = ly1.forward(spk[bi], idx[bi])
        z2 = ly2.forward(bs, z1)

        prediction = torch.argmin(z2, dim=1)
        correct += prediction.eq(targets[bi])
    pass
    return correct, sn


def main():
    # preprocess()
    Spike_train()


if __name__ == "__main__":
    main()
