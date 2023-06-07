from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import wandb
from munch import Munch
import torch
import torch.nn.functional as F


class Trainer(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def train_epoch(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def test(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def log(self) -> None:
        raise NotImplementedError
    
    
class ClassificationTrainer(Trainer):
    def __init__(self, model:torch.nn.Module, optimiser:torch.optim.Optimizer,
                 train_loader:torch.utils.data.DataLoader,
                 test_loader:torch.utils.data.DataLoader,
                 device:torch.device, config:Munch
                ) -> None:
        super().__init__()
        self.model = model
        self.optimiser = optimiser
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        self.logger = wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            config=config.default,
            reinit=True
        )
        
    def train(self) -> None:
        for _ in range(self.config.default.epochs):
            self.train_epoch()
            test_info = self.test()
            self.log(**test_info)
            
        self.logger.finish()
    
    def train_epoch(self) -> None:
        for _, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimiser.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimiser.step()
            self.logger.log({"train loss": loss.item()})
    
    def test(self) -> None:
        self.model.eval()
    
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, 
                                             reduction='mean'
                                            ).item()
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = 100. * correct / len(self.test_loader.dataset)
        test_loss /= len(self.test_loader.dataset)

        self.model.train()

        return {'test acc': test_acc, 'test loss': test_loss}
    
    def log(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.logger.log({k: v})
