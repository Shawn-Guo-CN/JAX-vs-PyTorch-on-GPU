import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from torch_geometric.nn import SAGEConv
from munch import Munch

class LeNet(nn.Module):
    def __init__(self, num_classes:int=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.out_dim = num_classes

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x
    

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.gnn_convs = nn.ModuleList()
        
        self.gnn_convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.gnn_convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        self.gnn_convs.append(SAGEConv(hidden_channels, num_classes))
            
    def reset_parameters(self):
        for conv_layer in self.gnn_convs:
            conv_layer.reset_parameters()
            
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  
            x = self.gnn_convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                
        return x.log_softmax(dim=-1)


BENCHMARK2MODEL = {
    'MNIST': LeNet,
    'CIFAR10': models.resnet18
}

GNN_BENCHMARKMODEL = {
    'CITESEER': SAGE,
    'REDDIT': SAGE
}


def get_model(config:Munch, device:torch.device) -> torch.nn.Module:
    benchmark = config.default.benchmark
    if config.default.benchmark in BENCHMARK2MODEL.keys():
        model_class = BENCHMARK2MODEL[benchmark]
        return model_class().to(device)
    elif benchmark in GNN_BENCHMARKMODEL.keys():
        model_class = GNN_BENCHMARKMODEL[benchmark]
        return model_class(config.default.in_channels,
                           config.default.hidden_channels,
                           config.default.num_layers,
                           config.default.num_classes).to(device)
    else:
        raise ValueError(f'No implemented model for {benchmark}')

