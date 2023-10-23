import os, argparse
import torch
from pdearena import utils
from pdearena.data.datamodule import PDEDataModule
from pdearena.lr_scheduler import LinearWarmupCosineAnnealingLR  # noqa: F401
from pdearena.models.pdemodel import PDEModel

def get_model():

    opt = {}
    opt['data'] = {}
    opt['data']['task'] = 'NavierStokes2D'
    opt['data']['data_dir'] = '/ibex/ai/home/liux0t/NavierStokes-2D'
    opt['data']['time_history'] = 4
    opt['data']['time_future'] = 1
    opt['data']['time_gap'] = 0
    opt['data']['pde'] = {}
    opt['data']['pde']['n_scalar_components'] = 1
    opt['data']['pde']['n_vector_components'] = 0
    opt['data']['pde']['trajlen'] = 14
    opt['data']['pde']['n_spatial_dim'] = 2
    opt['data']['batch_size'] = 8
    opt['data']['pin_memory'] = True
    opt['data']['num_workers'] = 1
    opt['data']['train_limit_trajectories'] = -1
    opt['data']['valid_limit_trajectories'] = -1
    opt['data']['test_limit_trajectories'] = -1

    opt['model'] = {}
    opt['model']['name'] = 'Unetmod-64'
    opt['model']['lr'] = 2e-4
    opt['model']['time_history'] = 1
    opt['model']['time_future'] = 1
    opt['model']['time_gap'] = 0
    opt['model']['max_num_steps'] = 5
    opt['model']['activation'] = "gelu"
    opt['model']['criterion'] = "mse"


    opt['data']['pde'] = argparse.Namespace(**opt['data']['pde'])

    model = PDEModel(**opt['model'], pdeconfig=opt['data']['pde'])
    print(model.model)

    return model.model

def get_data_loader():

    opt = {}
    opt['data'] = {}
    opt['data']['task'] = 'NavierStokes2D'
    opt['data']['data_dir'] = '/ibex/ai/home/liux0t/NavierStokes-2D'
    opt['data']['time_history'] = 4
    opt['data']['time_future'] = 1
    opt['data']['time_gap'] = 0
    opt['data']['pde'] = {}
    opt['data']['pde']['n_scalar_components'] = 1
    opt['data']['pde']['n_vector_components'] = 0
    opt['data']['pde']['trajlen'] = 14
    opt['data']['pde']['n_spatial_dim'] = 2
    opt['data']['batch_size'] = 8
    opt['data']['pin_memory'] = True
    opt['data']['num_workers'] = 1
    opt['data']['train_limit_trajectories'] = -1
    opt['data']['valid_limit_trajectories'] = -1
    opt['data']['test_limit_trajectories'] = -1

    opt['model'] = {}
    opt['model']['name'] = 'Unetmod-64'
    opt['model']['lr'] = 2e-4
    opt['model']['time_history'] = 1
    opt['model']['time_future'] = 1
    opt['model']['time_gap'] = 0
    opt['model']['max_num_steps'] = 5
    opt['model']['activation'] = "gelu"
    opt['model']['criterion'] = "mse"

    datamodule = PDEDataModule(**opt['data'])
    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()
    valid_dataloader, _ = datamodule.val_dataloader()
    test_dataloader, _ = datamodule.test_dataloader()
    
if __name__ == "__main__":
    model = get_model()
    x = torch.randn(10, 1, 128, 128)
    y = model(x)
    print(y.shape)