from typing import Tuple

import torch
from munch import Munch
from torchvision import transforms, datasets


def get_dataloaders(config:Munch) -> \
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
