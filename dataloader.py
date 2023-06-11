from typing import Tuple

import torch
from munch import Munch
from torchvision import transforms, datasets

from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.loader import DataLoader, NeighborSampler

def get_data(config:Munch):
    if config.default.benchmark in ['CITESEER']:
        dataset = Planetoid(root=config.default.data_dir, name=config.default.benchmark, split='full')
        data = dataset[0]
        train_mask = data.train_mask^data.val_mask
        test_mask = data.test_mask
    elif config.default.benchmark in ['REDDIT']:
        dataset = Reddit(root=config.default.data_dir+'/REDDIT')
        data = dataset[0]
        train_mask = data.train_mask^data.val_mask
        test_mask = data.test_mask
    elif config.default.benchmark in ['PRODUCT']:
        pass
    return data, train_mask, test_mask

def get_dataloaders(config:Munch) -> \
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config.default.benchmark in ['MNIST', 'CIFAR10']:
        train_set = eval('datasets.' + config.default.benchmark)(
            root=config.default.data_dir,
            train=True,
            download=True,
            transform=eval(config.default.train_transform)
        )
        
        test_set = eval('datasets.' + config.default.benchmark)(
            root=config.default.data_dir,
            train=False,
            download=True,
            transform=eval(config.default.test_transform)
        )
        
        train_kwargs = {'batch_size': config.default.batch_size,
                        'num_workers': config.torch.num_workers,
                        'pin_memory': config.torch.pin_memory,
                        'shuffle': True
                    }
        test_kwargs = train_kwargs.copy()
        test_kwargs.update({'batch_size': config.default.test_batch_size})
        
        train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
        
        return train_loader, test_loader
        
    elif config.default.benchmark in ['CITESEER', 'REDDIT']:
        data, train_mask, test_mask = get_data(config)
        
        train_loader = NeighborSampler(edge_index=data.edge_index, node_idx=train_mask, 
                                    batch_size=config.default.batch_size, 
                                    sizes=config.default.sizes)
        
        test_loader = NeighborSampler(edge_index=data.edge_index, node_idx=test_mask, 
                                    batch_size=config.default.batch_size, 
                                    sizes=config.default.sizes)

        return train_loader, test_loader, data.x, data.y
    
    else:
        raise ValueError(
            f'No implemented data loader for {config.default.benchmark}'
        )
    
    

