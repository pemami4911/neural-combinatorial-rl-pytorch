import math

import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """ implementation of the Graph Convolutional Layer from 
     https://arxiv.org/pdf/1609.02907.pdf """
    def __init__(self, input_dim, output_dim, dropout=0.0, use_cuda=False):
        weights_ = torch.FloatTensor(input_dim, output_dim)
        bias_ = torch.zeros(output_dim)

        if use_cuda: 
            weights_ = weights_.cuda()
            bias_ = bias_.cuda()

        self.weights = nn.Parameter(weights_)
        self.bias = nn.Parameter(bias_)
        self.dropout = nn.Dropout(dropout)

        self.weights.data.uniform_(-(1. / math.sqrt(output_dim)),
            1. / math.sqrt(output_dim))

    def forward(self, x):
        x = self.dropout(x)
        #TODO

