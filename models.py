import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv,SAGEConv

class GCN(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        self.conv1=GCNConv(in_dim,hidden_dim)
        self.conv2=GCNConv(hidden_dim,hidden_dim)
        self.classifier=(nn.Linear(hidden_dim,out_dim))

    def forward(self,x,edge_index):
        h=self.conv1(x,edge_index).relu()
        h=self.conv2(h,edge_index).relu()
        out=self.classifier(h).squeeze()
        return out

class NN(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        self.lin1=nn.Linear(in_dim,hidden_dim)
        self.lin2=nn.Linear(hidden_dim,hidden_dim)
        self.classifier=nn.Linear(hidden_dim,out_dim)

    def forward(self,x):
        h=self.lin1(x).relu()
        h=self.lin2(h).relu()
        out=self.classifier(h).squeeze()
        return out

class GCN_EW(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,edge_index):
        super().__init__()

        self.edge_weight=torch.nn.Parameter(torch.zeros(edge_index.shape[1]))

        self.conv1=GCNConv(in_dim,hidden_dim)
        self.conv2=GCNConv(hidden_dim,hidden_dim)
        self.classifier=nn.Linear(hidden_dim,out_dim)

    def forward(self,x,edge_index):
        h=self.conv1(x,edge_index,torch.exp(self.edge_weight)).relu()
        h=self.conv2(h,edge_index,torch.exp(self.edge_weight)).relu()
        out=self.classifier(h).squeeze()
        return out

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads)
        self.conv2 = GATConv(heads * hidden_dim, hidden_dim, heads)
        self.classifier = nn.Linear(heads * hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        out = self.classifier(h).squeeze()
        return out


class GSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
 
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.classifier = nn.Linear(hidden_dim, out_dim)
 
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        out = self.classifier(h).squeeze()
        return out
 
    def get_node_embeddings(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return h 


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_p=0.2):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')

        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)

        h = self.conv2(h, edge_index).relu()
        h = self.dropout(h)

        h = self.conv3(h, edge_index).relu()
        h = self.dropout(h)

        out = self.classifier(h).squeeze()
        return out

    def get_node_embeddings(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()
        return h  