import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math, os
import numpy as np

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Param:
        in_features, out_features, bias
    Input:
        features: N x C (n = # nodes), C = in_features
        adj: adjacency matrix (N x N)
    """

    def __init__(self, in_features, out_features, mat_path, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        #self.adj = Parameter(torch.Tensor(12, 12))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        adj_mat = np.load(mat_path)
        # add self-connection
        adj_mat = adj_mat + np.identity(adj_mat.shape[0], dtype=adj_mat.dtype)
        self.register_buffer('adj', torch.from_numpy(adj_mat))
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        # stdv = 1. / math.sqrt(self.adj.size(1))
        # self.adj.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(self.adj.unsqueeze(0).repeat(b, 1, 1), support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCN, self).__init__()
        # two layer GCN, TODO: try single layer & pre-defined adjacency matrix
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.bn1 = nn.BatchNorm1d(nhid)
        # self.gc2 = GraphConvolution(nhid, nout, mat_path)
        # self.bn2 = nn.BatchNorm1d(nout)
        # self.dropout = dropout

    def forward(self, x):
        
        x = self.gc1(x)
        x = x.transpose(1, 2).contiguous()
        x = self.bn1(x).transpose(1, 2).contiguous()
        x = F.relu(x)
        
        # x = F.dropout(x, self.dropout, training=self.training)
        
        # x = self.gc2(x)
        
        # x = x.transpose(1, 2).contiguous()
        # x = self.bn2(x).transpose(1, 2).contiguous()
        # x = F.relu(x)
        
        # x = F.relu(self.gc2(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x
    
class PEM(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        mat_path = os.path.join(
            'assets',
            '{}.npy'.format(opt['dataset'])
        )
        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))
        #self.graph_embedding = torch.nn.Sequential(GCN(2, 32, 32, mat_path))
        in_dim = 192#24

        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            # # receptive filed: 7
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )
        # 0:micro(start,end,None),    3:macro(start,end,None),
        # 6:micro_apex,7:macro_apex,  8:micro_action, macro_action
        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b*t, n, c)  # (b*t, n, c)
        x = self.graph_embedding(x).reshape(b, t, -1).transpose(1, 2)   # (b, C=384=12*32, t)
        #x = self.graph_embedding(x).reshape(b, t, n, 16)
        x = self._sequential(x)
        x = self._classification(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    import yaml
    # load config & params.
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    
    x = torch.randn((16, 64, 12, 2))        # (b, t, n, c)
    model = PEM(opt)
    
    out = model(x)
    print(out.shape)
    