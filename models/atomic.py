import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, LayerNorm, Sequential, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge, scatter

from torch_scatter import scatter_softmax

class LocationBasedAttentionReadout(nn.Module):
    def __init__(self, in_channels):
        super(LocationBasedAttentionReadout, self).__init__()
        self.W_a = nn.Parameter(torch.Tensor(in_channels, 1))
        nn.init.xavier_uniform_(self.W_a)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):

        importance_score = torch.matmul(x, self.W_a).squeeze(-1)
        attn_score = scatter_softmax(importance_score, batch, dim=0)
        
        ## weighted sum
        x_weighted = x * attn_score.unsqueeze(-1)
        readout = scatter(x_weighted, batch, dim=0, reduce='sum')

        return readout, attn_score
    
class ATOMIC(nn.Module):
    def __init__(self, 
                 input_dim, hidden_dim, output_dim, drop_rate, num_layers, num_heads, clf_hidden_dim, clf_drop_rate,
                 preinitialized_dnabert_embeddings, preinitialized_random_fea_embeddings,
                 lambda_scale_abn, lambda_scale_dnabert, init_embed_dim): 
        super().__init__()

        convs = []
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.output_dim = output_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_drop_rate = clf_drop_rate
        self.lambda_scale_abn = lambda_scale_abn
        self.lambda_scale_dnabert = lambda_scale_dnabert

        for i in range(self.num_layers):
            if i == 0:
                ## first layer
                convs.append((GATv2Conv(in_channels = input_dim, out_channels = hidden_dim, heads = self.num_heads,
                                        concat = True, residual = True,
                                        dropout = self.drop_rate
                                        ), 'x, edge_index -> x'))
                convs.append(LayerNorm(hidden_dim * self.num_heads, mode='graph'))
                convs.append(nn.Mish())
            elif i != self.num_layers - 1:
                convs.append((GATv2Conv(in_channels = hidden_dim * self.num_heads, out_channels = hidden_dim, heads = self.num_heads, 
                                        concat = True, residual = True, 
                                        dropout = self.drop_rate
                                        ), 'x, edge_index -> x'))
                
                convs.append(LayerNorm(hidden_dim * self.num_heads, mode='graph'))
                convs.append(nn.Mish())
            else: 
                ## last layer
                convs.append((GATv2Conv(in_channels = hidden_dim * self.num_heads, out_channels = output_dim, heads = 1, 
                                        concat = False, residual = True,
                                        dropout = self.drop_rate
                                        ), 'x, edge_index -> x'))
                convs.append(LayerNorm(output_dim, mode='graph'))
                convs.append(nn.Mish())

        self.convs = Sequential('x, edge_index', convs)
        
        self.emb_mlp = nn.Sequential(
            nn.Linear(init_embed_dim + 64, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Mish(),
        )

        self.lin1 = nn.Sequential(
            nn.Linear(output_dim, clf_hidden_dim),
            nn.BatchNorm1d(num_features = clf_hidden_dim),
            nn.Mish(),
            nn.Dropout(p = self.clf_drop_rate)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(clf_hidden_dim, clf_hidden_dim // 2),
            nn.BatchNorm1d(num_features = clf_hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(p = self.clf_drop_rate),
        )
        self.output_lin = nn.Sequential(
            nn.Linear(clf_hidden_dim // 2 , 1),
            nn.Sigmoid()
        )

        self.dnabert_embeddings = nn.Embedding.from_pretrained(preinitialized_dnabert_embeddings, freeze=False)
        self.random_fea_embeddings = nn.Embedding.from_pretrained(preinitialized_random_fea_embeddings, freeze=False)

        self.attn_readout = LocationBasedAttentionReadout(output_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the all layer"""
        for name, p in self.named_parameters():
            if 'embedding' not in name:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


    def forward(self, x, edge_index, batch, x_ids):

        dnabert_embed = self.dnabert_embeddings(x_ids.squeeze())
        random_fea_embed = self.random_fea_embeddings(x_ids.squeeze())
        dnabert_embed = self.lambda_scale_dnabert * dnabert_embed

        x = self.lambda_scale_abn * x
        x = x * random_fea_embed
        x = torch.concat([x, dnabert_embed], axis = 1)

        ## x embedding
        x = self.emb_mlp(x)

        attn_weights = []
        for conv in self.convs:
            if isinstance(conv, GATv2Conv):
                x, attn_weights = conv(x, dropout_edge(edge_index, p = self.clf_drop_rate, training = self.training)[0], return_attention_weights = True)
            elif isinstance(conv, LayerNorm):
                x = conv(x, batch)
            else:
                x = conv(x)
                x = F.dropout(x, p = self.clf_drop_rate, training = self.training)
        
        graph_x, pooling_attn_scores = self.attn_readout(x, batch)

        x = self.lin1(graph_x)
        x = self.lin2(x)
        x = self.output_lin(x)

        return x, graph_x, attn_weights, pooling_attn_scores
