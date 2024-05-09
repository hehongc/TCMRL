import torch
from torch import nn as nn
from torch.nn import functional as F
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule

from torch.nn.utils import weight_norm
import torch.nn.init as init
import pywt
import numpy as np
import numpy

from rlkit.torch.networks import Mlp


class SelfAttnEncoder(PyTorchModule):
    def __init__(self,
                 input_dim,
                 num_output_mlp=0,
                 task_gt_dim=5,
                 init_w=3e-3,
                 ):
        super(SelfAttnEncoder, self).__init__()

        self.input_dim = input_dim
        self.score_func = nn.Linear(input_dim, 1)

        self.num_output_mlp = num_output_mlp

        if num_output_mlp > 0:
            self.output_mlp = Mlp(
                input_size=input_dim,
                output_size=task_gt_dim,
                hidden_sizes=[200 for i in range(num_output_mlp - 1)]
            )

        self.score_func.weight.data.uniform_(-init_w, init_w)
        self.score_func.bias.data.uniform_(-init_w, init_w)


    def forward(self, input, z_mean):

        b, N, dim = input.shape

        z_mean = [z.repeat(N, 1) for z in z_mean]
        z_mean = torch.cat(z_mean, dim=0)

        score_func_input_tuple_representations = input.reshape(-1, dim)
        score_func_input_z_mean = z_mean.reshape(score_func_input_tuple_representations.shape[0], -1)
        score_func_input = torch.cat([score_func_input_tuple_representations, score_func_input_z_mean], dim=-1)


        scores = self.score_func(score_func_input).reshape(b, N)
        scores_before_softmax = scores
        scores_sigmoid = F.sigmoid(scores)

        scores = F.softmax(scores, dim=-1)

        reverse_scores = 1 - scores


        context = scores.unsqueeze(-1).expand_as(input).mul(input)
        context_sum = context.sum(1)

        reverse_context = reverse_scores.unsqueeze(-1).expand_as(input).mul(input)
        reverse_context_sum = reverse_context.sum(1)

        return context, context_sum, scores, scores_sigmoid, reverse_context, reverse_context_sum



class CVAE(nn.Module):
    def __init__(self,
                 hidden_size=64,
                 num_hidden_layers=1,
                 z_dim=20,
                 action_dim=5,
                 state_dim=2,
                 reward_dim=1,
                 use_ib=False,
                 ):
        
        super(CVAE, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.z_dim = z_dim

        self.use_ib = use_ib

        if self.use_ib:
            self.encoder = Mlp(
                input_size=self.state_dim*2+self.action_dim+self.reward_dim,
                output_size=self.z_dim*2,
                hidden_sizes=[hidden_size for i in  range(num_hidden_layers)]
            )
        else:
            self.encoder = Mlp(
                input_size=self.state_dim * 2 + self.action_dim + self.reward_dim,
                output_size=self.z_dim,
                hidden_sizes=[hidden_size for i in range(num_hidden_layers)]
            )

        self.decoder = Mlp(
            input_size=self.z_dim+self.state_dim+self.action_dim,
            output_size=self.state_dim+self.reward_dim,
            hidden_sizes=[hidden_size for i in range(num_hidden_layers)]
        )




class SpatialAttention(PyTorchModule):
    def __init__(self, num_decompose_parts, input_dim, time_steps):
        super(SpatialAttention, self).__init__()

        self.num_decompose_parts = num_decompose_parts
        self.input_dim = input_dim
        self.time_steps = time_steps

        self.w1 = nn.Linear(self.time_steps, 1, bias=False)
        self.w2 = nn.Linear(self.in_features, self.time_steps, bias=False)
        self.w3 = nn.Linear(self.in_features, 1, bias=False)
        self.bs = nn.Parameter(torch.randn((self.num_decompose_parts, self.num_decompose_parts)))
        self.Vs = nn.Parameter(torch.randn((self.num_decompose_parts, self.num_decompose_parts)))

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.w1.weight)
        init.xavier_uniform_(self.w2.weight)
        init.xavier_uniform_(self.w3.weight)
        init.xavier_uniform_(self.bs)
        init.xavier_uniform_(self.Vs)



    def forward(self, x):
        tmp1 = self.w1(x.reshape(-1, self.time_steps)) # tmp1 shape (num_decompose_parts*in_features/contexts, 1)
        tmp2 = self.w2(tmp1.reshape(-1, self.input_dim)) # shape (num_decompose_parts, time_steps)
        tmp3 = self.w3(x.permute(0, 2, 1).reshape(-1, self.input_dim)) # shape (num_decompose_parts * time_step, 1)
        tmp3 = tmp3.reshape(-1, self.time_steps).T  # shape (time_steps, num_decompose_parts)
        tmp4 = th.matmul(tmp2.reshape(-1, self.num_decompose_parts, self.time_steps), tmp3.reshape(-1, self.time_steps, self.num_decompose_parts)).reshape(-1, self.num_decompose_parts, self.num_decompose_parts) # shape (num_decompose_parts, num_decompose_parts)
        S = torch.nn.functional.sigmoid(tmp4 + self.bs) * self.Vs # shape (batch_size, n_stocks, n_stocks)
        S_normalized = torch.nn.functional.softmax(S, dim=2) # normalize by rows
        return S_normalized


def series_decomposition(data, level=3):
    '''
        reference: https://www.freesion.com/article/2882783957/

        decompose the close price series into multi-level series
        using haar decomposition
        input param: data.shape=[lookback, stocks_num], max decompose level
        output param: decomposed array
        '''
    data = np.array(data.cpu())
    dec_list = [[] for i in range(level + 1)]
    wavelet = 'haar'
    for i in range(data.shape[0]):
        coeffs = pywt.wavedec(data[i], wavelet, level=level)
        level_id = np.eye(level + 1)
        for j, coeff in enumerate(coeffs):
            if level == 1:
                rec_coefs = []
                level_id_list = [[item] for item in level_id[j]]
                temp_coefs = np.multiply(coeffs, level_id_list).tolist()
                for coef in temp_coefs:
                    rec_coefs.append(np.array(coef))
            else:
                rec_coefs = np.multiply(coeffs, level_id[j]).tolist()
            temp = pywt.waverec(rec_coefs, wavelet)
            dec_list[j].append(temp.astype(np.float32))

    return torch.tensor(dec_list)