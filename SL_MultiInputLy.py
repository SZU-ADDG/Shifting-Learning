import torch


class SNN_MInLy:
    def __init__(self, inCh, outCh, e_ts=None, e_tp=None):
        self.inCh, self.outCh = inCh, outCh
        self.th = 1.0
        if e_ts is None:  # exponential shifting timings
            self.e_ts = torch.rand(self.inCh, self.outCh) * (8.0 / self.inCh)  # initialized by uniform distribution
        else:
            self.e_ts = e_ts
        if e_tp is None:  # exponential passing time between neurons added to the output
            self.e_tp = torch.ones(outCh)
        else:
            self.e_tp = e_tp
        self.cause_mask = torch.tensor(0)  # casual set record
        # Adam related parameters
        self.b1, self.b2, self.ep = 0.9, 0.9, 1e-8
        self.t, self.adam_m_Ets, self.adam_v_Ets = 0, torch.zeros_like(self.e_ts), torch.zeros_like(self.e_ts)
        self.adam_m_Etp, self.adam_v_Etp = torch.zeros_like(self.e_tp), torch.zeros_like(self.e_tp)

    def forward(self, spk, idx, dv=torch.device("cpu")):  # IF
        inCh, outCh = self.inCh, self.outCh
        inNum = spk.size()[0]
        spkEx = torch.tile(torch.reshape(spk, [inNum, 1]), [1, outCh])  # (inNum, outCh) in
        idxEx = torch.tile(torch.reshape(idx, [inNum, 1]), [1, outCh])
        EtsEx = torch.gather(self.e_ts, dim=0, index=idxEx)  # (inNum, outCh) wt
        z_inEx = torch.exp(spkEx)
        EtsZinMul = EtsEx * z_inEx

        EtsSum = torch.cumsum(EtsEx, dim=0)
        EtsZinMulSum = torch.cumsum(EtsZinMul, dim=0)
        z_outAll = EtsZinMulSum / torch.clamp(EtsSum - self.th, 1e-10, 1e10)
        z_outAll = torch.where(EtsSum < self.th, 1e5 * torch.ones_like(z_outAll), z_outAll)
        z_outAll = torch.where(z_outAll < z_inEx, 1e5 * torch.ones_like(z_outAll), z_outAll)
        z_out, z_outIdx = torch.min(z_outAll, dim=0)

        # Complete casual set record
        z_outIdxEx = torch.tile(torch.reshape(z_outIdx, [1, outCh]), [inNum, 1])
        locEx = torch.tile(torch.reshape(torch.arange(inNum), [inNum, 1]), [1, outCh]).to(dv)
        self.cause_mask = torch.where(locEx <= z_outIdxEx, 1, 0)
        return torch.reshape(z_out * self.e_tp, [1, outCh])

    def backward(self, delta, spk, idx, z_out, lr_Ets, lr_Etp):
        inCh, outCh = self.inCh, self.outCh
        inNum = spk.size()[0]
        e_tpEx = torch.tile(torch.reshape(torch.exp(self.e_tp), [1, outCh]), [inNum, 1])
        deltaEx = torch.tile(torch.reshape(delta, [1, outCh]), [inNum, 1])
        z_inEx = torch.tile(torch.reshape(torch.exp(spk), [inNum, 1]), [1, outCh])
        z_outEx = torch.tile(torch.reshape(z_out, [1, outCh]), [inNum, 1])
        T = (z_inEx - z_outEx) * self.cause_mask  # (4367, 800)

        adj_Ets_Mul = deltaEx * e_tpEx * T
        adj_Ets = torch.zeros((inCh, outCh)).to(torch.float64)
        adj_Ets = adj_Ets.index_add(0, idx, adj_Ets_Mul)
        adj_Etp = delta * z_out
        # print(inNum)

        # Adam
        self.t += 1
        self.adam_m_Ets = self.b1 * self.adam_m_Ets + (1.0 - self.b1) * adj_Ets
        self.adam_v_Ets = self.b2 * self.adam_v_Ets + (1.0 - self.b2) * adj_Ets * adj_Ets
        M_Ets = self.adam_m_Ets / (1.0 - self.b1 ** self.t)
        V_Ets = self.adam_v_Ets / (1.0 - self.b2 ** self.t)
        self.e_ts -= lr_Ets * (M_Ets / (torch.sqrt(V_Ets) + self.ep))

        self.adam_m_Etp = self.b1 * self.adam_m_Etp + (1.0 - self.b1) * adj_Etp
        self.adam_v_Etp = self.b2 * self.adam_v_Etp + (1.0 - self.b2) * adj_Etp * adj_Etp
        M_Etp = self.adam_m_Etp / (1.0 - self.b1 ** self.t)
        V_Etp = self.adam_v_Etp / (1.0 - self.b2 ** self.t)
        self.e_tp = self.e_tp - lr_Etp * (M_Etp / (torch.sqrt(V_Etp) + self.ep))
