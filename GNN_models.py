import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch_geometric.nn.inits import glorot
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import softmax as scatter_softmax
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import ASAPooling, TopKPooling, SAGPooling


class GNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer=1, agg_method='SUM',
                 gnn_type='GCN', softmax_temp=1., use_linear_layer=True):

        super(GNN, self).__init__()
        self.layer = layer
        self.gnn_type = gnn_type
        self.agg_method = agg_method
        self.conv_layers = nn.ModuleList()
        
        self.softmax_temp = softmax_temp
        self.use_linear_layer = use_linear_layer

        if gnn_type == 'GCN':
            self.conv_layers.append(GCNConv(in_dim, hid_dim))
            for i in range(1, layer):
                self.conv_layers.append(GCNConv(hid_dim, hid_dim))

        elif gnn_type == 'GAT':
            self.conv_layers.append(GATConv(in_dim, hid_dim))
            for i in range(1, layer):
                self.conv_layers.append(GATConv(hid_dim, hid_dim))

        else:
            raise NameError('GNN Type Not Supported')

        if use_linear_layer:
            self.attn_pool = nn.Parameter(torch.Tensor(hid_dim, 1))
        else:
            self.attn_pool = nn.Parameter(torch.Tensor(out_dim, 1))
        
        if use_linear_layer:
            self.lin = Linear(hid_dim, out_dim)
        else:
            if self.layer == 1:
                if gnn_type == 'GCN':
                    self.conv_layers[-1] = GCNConv(in_dim, out_dim)
                if gnn_type == 'GAT':
                    self.conv_layers[-1] = GATConv(in_dim, out_dim)
            else:
                if gnn_type == 'GCN':
                    self.conv_layers[-1] = GCNConv(hid_dim, out_dim)
                if gnn_type == 'GAT':
                    self.conv_layers[-1] = GATConv(hid_dim, out_dim)


        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        r"""Resets all learnable parameters of the module."""

        for module in self.conv_layers:
            module.reset_parameters()

        if self.use_linear_layer:
            self.lin.reset_parameters()

        glorot(self.attn_pool)


    def forward(self, x, edge_index, batch, ptr=None, ATTN_RETURN_DETAILS=True):
        for i in range(0, self.layer):
            x = self.conv_layers[i](x, edge_index)

            if i != self.layer-1:  ## DON'T TAKE RELU in the last layer
                x = F.relu(x)

        if self.agg_method == 'SUM':
            x = global_add_pool(x, batch=batch)

        elif self.agg_method == 'AVG':
            x = global_mean_pool(x, batch=batch)

        elif self.agg_method == 'MAX':
            x = global_max_pool(x, batch=batch)

        elif self.agg_method == 'ATTN':
            x = x * self.softmax_temp
            alpha = torch.matmul(x, self.attn_pool)
            alpha = scatter_softmax(src=alpha, ptr=ptr)
            x = x * alpha
            x = global_add_pool(x,  batch=batch)

        x = F.dropout(x, p=0.2, training=self.training)

        if self.use_linear_layer:
            x = self.lin(x)

        # if self.agg_method == 'ATTN' and ATTN_RETURN_DETAILS:
        #     return (x, alpha, self.attn_pool)

        return x


class GNN_ClusterPooling(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer=1, agg_method='ASAP', global_pooling_method='MAX', gnn_type='GCN'):
        super(GNN_ClusterPooling, self).__init__()
        self.layer = layer
        self.gnn_type = gnn_type
        self.agg_method = agg_method
        self.global_pooling_method = global_pooling_method
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.attn_pool = nn.Parameter(torch.Tensor(hid_dim, 1))

        if gnn_type == 'GCN':
            self.conv_layers.append(GCNConv(in_dim, hid_dim))
            for i in range(1, layer):
                self.conv_layers.append(GCNConv(hid_dim, hid_dim))

        elif gnn_type == 'GAT':
            self.conv_layers.append(GATConv(in_dim, hid_dim))
            for i in range(1, layer):
                self.conv_layers.append(GATConv(hid_dim, hid_dim))
        else:
            raise NameError('GCN Type Not Supported')


        if agg_method == 'ASAP':
            for i in range(0, layer):
                self.pool_layers.append(ASAPooling(hid_dim))

        elif agg_method == 'TOPK':
            for i in range(0, layer):
                self.pool_layers.append(TopKPooling(hid_dim))

        elif agg_method == 'SAGP':
            for i in range(0, layer):
                self.pool_layers.append(SAGPooling(hid_dim))
        else:
            raise NameError('Pooling Method Type Not Supported')

        self.lin = Linear(hid_dim, out_dim)
        self.reset_parameters()


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for module in self.conv_layers:
            module.reset_parameters()
        for pool in self.pool_layers:
            pool.reset_parameters()
        glorot(self.attn_pool)


    def forward(self, x, edge_index, batch, ptr=None):

        for i in range(0, self.layer):
            x = self.conv_layers[i](x, edge_index)
            pooling_output = self.pool_layers[i](x=x, edge_index=edge_index, batch=batch)
            x, edge_index, batch = pooling_output[0], pooling_output[1], pooling_output[3]
            x = F.relu(x)

        if self.global_pooling_method == 'SUM':
            x = global_add_pool(x, batch=batch)

        elif self.global_pooling_method == 'AVG':
            x = global_mean_pool(x, batch=batch)

        elif self.global_pooling_method == 'MAX':
            x = global_max_pool(x, batch=batch)

        elif self.global_pooling_method == 'ATTN':
            alpha = torch.matmul(x, self.attn_pool)
            ptr = torch._convert_indices_from_coo_to_csr(batch, int(batch.max())+1)
            alpha = scatter_softmax(src=alpha, ptr=ptr)
            x = x * alpha
            x = global_add_pool(x, batch=batch)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        return x


def GNN_Model_Wrapper(in_dim, hid_dim, out_dim, layer=1, agg_method='ASAP', global_pooling_method='MAX',
                      gnn_type='GCN', softmax_temp=1., use_linear_layer=True):

    if agg_method in ['SAGP', 'ASAP', 'TOPK']:
        return GNN_ClusterPooling(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, layer=layer, agg_method=agg_method,
                                  global_pooling_method=global_pooling_method, gnn_type=gnn_type)

    elif agg_method in ['ATTN', 'SUM', 'MAX', 'AVG']:
        return GNN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, layer=layer, agg_method=agg_method,
                   gnn_type=gnn_type, softmax_temp=softmax_temp, use_linear_layer=use_linear_layer)

    else:
        raise NotImplementedError

