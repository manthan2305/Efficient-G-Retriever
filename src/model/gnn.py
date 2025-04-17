import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv, BatchNorm
from torch.nn import LayerNorm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr

class GraphTransformer2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GraphTransformer2, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()  # LayerNorm for residual connections

        # Edge Feature Encoder
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )

        # Input Layer
        self.convs.append(TransformerConv(in_channels=in_channels, 
                                          out_channels=hidden_channels // num_heads, 
                                          heads=num_heads, 
                                          edge_dim=hidden_channels, 
                                          dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))
        self.lns.append(LayerNorm(hidden_channels))

        # Hidden Layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, 
                                              out_channels=hidden_channels // num_heads, 
                                              heads=num_heads, 
                                              edge_dim=hidden_channels, 
                                              dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels))
            self.lns.append(LayerNorm(hidden_channels))

        # Output Layer
        self.convs.append(TransformerConv(in_channels=hidden_channels, 
                                          out_channels=out_channels // num_heads, 
                                          heads=num_heads, 
                                          edge_dim=hidden_channels, 
                                          dropout=dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for layer in self.edge_encoder:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x, adj_t, edge_attr):

        # Encode edge features
        edge_attr = self.edge_encoder(edge_attr)

        # Calculate Relative Positional Encoding
        # Here, adj_t is the edge_index
        row, col = adj_t
        relative_pos = (x[row] - x[col]).norm(p=2, dim=-1, keepdim=True)
        
        # Normalize positional encodings
        relative_pos = (relative_pos - relative_pos.mean()) / (relative_pos.std() + 1e-6)
        
        # Add positional encodings to edge features
        edge_attr = edge_attr + relative_pos

        # Forward Pass through Transformer Layers
        for i, conv in enumerate(self.convs[:-1]):
            # Save input for residual connection
            residual = x
            
            # Graph Transformer Layer
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            
            # Batch Normalization
            x = self.bns[i](x)
            
            # Activation
            x = F.relu(x)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual Connection and Layer Normalization
            x = x + residual
            x = self.lns[i](x)
        
        # Final Layer (No activation, no residual connection)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        
        return x, edge_attr

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x,edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gt': GraphTransformer,
    'gt2': GraphTransformer2,
}
