from __future__ import absolute_import
import torch.nn as nn
from .sem_graph_conv import SemGraphConv


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class SimpleSemGCN(nn.Module):
    def __init__(self, adj, dim, num_layers=4):
        super(SimpleSemGCN, self).__init__()

        _gconv_layers = []
        for i in range(num_layers):
            _gconv_layers.append(_GraphConv(adj, dim, dim))

        self.gconv_layers = nn.Sequential(*_gconv_layers)

    def forward(self, x):
        out = self.gconv_layers(x)
        return out
