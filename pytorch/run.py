import sys
import argparse
import torch
import pathlib
from munch import Munch


PROJ_DIR = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(PROJ_DIR)
from utils import get_config, get_run_config
from dataloader import get_dataloaders
from pytorch.trainer import ClassificationTrainer, GNNClassificationTrainer
from pytorch.models import get_model


def train_on_benchmark(config:Munch) -> None:
    if config.default.benchmark in ['MNIST', 'CIFAR10']:
        device = torch.device('cuda') #TODO: add multi-gpu support
        train_loader, test_loader = get_dataloaders(config)
        model = get_model(config, device)
        optimiser = eval('torch.optim.' + config.default.optim)(
                         model.parameters(), 
                         lr=config.default.lr, 
                         momentum=config.default.momentum, 
                         weight_decay=config.default.weight_decay
                        )
        lr_scheduler = eval(
            'torch.optim.lr_scheduler.' + config.default.lr_scheduler
        )
        
        trainer = ClassificationTrainer(model, optimiser, lr_scheduler, 
                                        train_loader, test_loader, 
                                        device, config
                                       )
        trainer.train()
        
    elif config.default.benchmark in ['CITESEER', 'REDDIT']:
        device = torch.device('cuda')
        train_loader, test_loader, feature, y = get_dataloaders(config)
        model = get_model(config, device)
        
        optimiser = eval('torch.optim.' + config.default.optim)(
                         model.parameters(), 
                         lr=config.default.lr, 
                         momentum=config.default.momentum, 
                         weight_decay=config.default.weight_decay
                        )
        lr_scheduler = eval(
            'torch.optim.lr_scheduler.' + config.default.lr_scheduler
        )
        
        trainer = GNNClassificationTrainer(model, optimiser, lr_scheduler,
                                           feature, y,
                                           train_loader, test_loader,
                                           device, config
                                           )
        trainer.train()
        
    else:
        raise ValueError(
            f'No implemented trainer for {config.default.benchmark}'
        )


def per_sample_grad(config:Munch) -> None:
    pass


def main(args:argparse.Namespace) -> None:
    args.benchmark = args.benchmark.upper()
    gpu_name = torch.cuda.get_device_name(0)
    args.gpu = gpu_name
    
    config = get_run_config(args)
    
    if args.use_case == 1:
        train_on_benchmark(config)
    elif args.use_case == 2:
        per_sample_grad(config)
    else:
        raise ValueError(f'Invalid use case: {args.use_case}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', type=str, default='MNIST')
    parser.add_argument('-u', '--use_case', type=int, default=1)
    parser.add_argument('-m', '--multi_gpu', action='store_true', default=False)
    
    main(parser.parse_args())