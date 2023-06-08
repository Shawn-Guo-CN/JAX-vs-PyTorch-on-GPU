import os
import pathlib
import toml
import argparse
from munch import Munch, munchify


def get_project_root() -> pathlib.Path:
    return str(pathlib.Path(__file__).parent.absolute())


def get_config() -> Munch:
    proj_dir = get_project_root()
    config = toml.load(os.path.join(proj_dir, 'config.toml'))
    config = munchify(config)
    return config


def get_run_config(args:argparse.Namespace) -> Munch:
    config = get_config()
    config.default.update(config[args.benchmark])
    config.default.update(vars(args))
    config.default.update({
        'proj_dir': get_project_root(),
        'data_dir': os.path.join(get_project_root(), 'data')
    })
    config.wandb.update({
        'name': f'{args.benchmark}_{args.use_case}_{args.gpu}',
    })
    return config