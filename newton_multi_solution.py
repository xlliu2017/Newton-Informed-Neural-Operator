import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MgNO_DC_5, MgNO_DC_6
from fno2d import FNO2d, SpectralDecoder
import os, logging
import numpy as np
import matplotlib.pyplot as plt

from utilities3 import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import argparse
torch.set_printoptions(threshold=100000)
# torch.set_default_tensor_type('torch.DoubleTensor')
    
class DeepONet(nn.Module):
    def __init__(self, branch_features, trunk_features, output_features, grid_size=63):
        super(DeepONet, self).__init__()
        self.grid_size = grid_size
        self.branch = nn.Sequential(
            # reshape the input to a 2D grid
            # then coarsely downsample the grid of 63x63 to 29x29
            # nn.Unflatten(1, (1, grid_size, grid_size)),
            nn.Conv2d(1, 128, kernel_size=7, stride=2),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 128),
            nn.GELU(),
            nn.Linear(128, branch_features),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, trunk_features),
            nn.GELU(),
            nn.Linear(trunk_features, trunk_features),
            nn.GELU(),
            nn.Linear(trunk_features, trunk_features),
        )
        grid = torch.linspace(0, 1, steps=grid_size)
        xv, yv = torch.meshgrid(grid, grid)
        grid = torch.stack((xv.flatten(), yv.flatten()), axis=1)
        self.register_buffer('grid', grid)

    
    
    def forward(self, x_branch):
        batch_size = x_branch.shape[0]
        branch_out = self.branch(x_branch)
        
        trunk_out = self.trunk(self.grid)
        # branch.shape is (batch_size, trunk_features), trunk.shape is (points_num, trunk_features)
        # the out should be (batch_size, points_num)
        out = torch.mm(branch_out, trunk_out.T)
        return out.view(batch_size, 1, self.grid_size, self.grid_size)

class DeepONet_POD(nn.Module):
    def __init__(self, branch_features, trunk_features, output_features, grid_size=63, V=None):
        super(DeepONet_POD, self).__init__()
        self.grid_size = grid_size
        self.branch = nn.Sequential(
            # reshape the input to a 2D grid
            # then coarsely downsample the grid of 63x63 to 29x29
            # nn.Unflatten(1, (1, grid_size, grid_size)),
            nn.Conv2d(2, 128, kernel_size=7, stride=2),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1,),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 128),
            nn.GELU(),
            nn.Linear(128, branch_features),
        )
        # self.branch_2 = nn.Sequential(
        #     nn.Conv2d(1,128, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(128,1, kernel_size=1)
        # )
        
        self.register_buffer('trunk', V[:branch_features,...].view(branch_features,-1))

    
    
    def forward(self, x_branch):
        batch_size = x_branch.shape[0]
        branch_out = self.branch(x_branch)
        out = torch.mm(branch_out, self.trunk)
        out = out.view(batch_size, 2, self.grid_size, self.grid_size) #+ self.branch_2(x_branch)
        return out

def objective(dataOpt, modelOpt, optimizerScheduler_args,
                tqdm_disable=True, 
              log_if=False, validate=False, model_type='MgNO_DC_5', 
              model_save=False, tune_if=False, test_mode=False,):
    
    ################################################################
    # configs
    ################################################################

    print(os.path.basename(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = getSavePath(dataOpt['data'], model_type)
    MODEL_PATH_PARA = getSavePath(dataOpt['data'], model_type, flag='para')
    if log_if:
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=MODEL_PATH,
                        filemode='w')
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(model_type)
        logging.info(f"dataOpt={dataOpt}")
        logging.info(f"modelOpt={modelOpt}")
        logging.info(f"optimizerScheduler_args={optimizerScheduler_args}")
  
    
    ################################################################
    # load data and data normalization
    ################################################################

    AS_list, deltaAS_list = torch.load('/home/liux0t/neural_MG/pytorch/newton_multiSolutions_test_only3.pt')
    V = torch.load('/home/liux0t/neural_MG/pytorch/newton_multiSolutions_test_only3_V.pt')
    V = V.float().contiguous().to(device)
    y = deltaAS_list.float().to(device) 
    x = AS_list.to(device).float()
    
  # take out AS_list a specific pattern as test set
    x_test = x[:54,...]
    y_test = y[:54,...]
    
    # randomly permute x_train
    perm = torch.randperm(x.size(0)-54)
    x = x[54:,...][perm]
    y = y[54:,...][perm]

    dataOpt['dataSize'] = {'train': range(10000,25000), 'test': range(25000, 28000), 'val':range(600,650), 'train_l2':range(10000)}
    x_train = x[dataOpt['dataSize']['train'],...]
    y_train = y[dataOpt['dataSize']['train'],...]
    x_test = torch.concatenate((x_test, x[dataOpt['dataSize']['test'],...]), dim=0)
    y_test = torch.concatenate((y_test, y[dataOpt['dataSize']['test'],...]), dim=0)
    
    x_train_l2 = x[dataOpt['dataSize']['train_l2'],...]
    y_train_l2 = y[dataOpt['dataSize']['train_l2'],...]
    # x_val = x[dataOpt['dataSize']['val'],...]
    # y_val = y[dataOpt['dataSize']['val'],...]

    # x_train_2 = x_2[dataOpt['dataSize']['train'],...]
    # y_train_2 = y_2[dataOpt['dataSize']['train'],...]
    # x_test_2 = x_2[dataOpt['dataSize']['test'],...]
    # y_test_2 = y_2[dataOpt['dataSize']['test'],...]

    # x_train = torch.cat((x_train, x_train_2), dim=0)
    # y_train = torch.cat((y_train, y_train_2), dim=0)
    # x_test = torch.cat((x_test, x_test_2), dim=0)
    # y_test = torch.cat((y_test, y_test_2), dim=0)
    

    if dataOpt['GN']:
        x_normalizer = GaussianNormalizer(x_train) #UnitGaussianNormalizer
        # x_train = x_normalizer.encode(x_train)
        # x_test = x_normalizer.encode(x_test)

    # if model_type == 'FNO':
    #     x_train = x_train.squeeze()
    #     x_test = x_test.squeeze()
    #     x_train = x_train[:, ..., np.newaxis]
    #     x_test = x_test[:, ..., np.newaxis]   
  

    if model_type in {'MgNO_DC_6', 'FNO'}:
        y_normalizer = GaussianNormalizer(y_train)
    
    # plot the data
    
    # visual2d(x_train[0, ..., 0].cpu().numpy(), '1')
    # visual2d(x_train[1, ..., 0].cpu().numpy(), '2')
    # visual2d(x_train[2, ..., 0].cpu().numpy(), '3')
    # visual2d(x_train[3, ..., 0].cpu().numpy(), '4')
    # visual2d(y_train[0, ...].cpu().numpy(), '5')
    # visual2d(y_train[1, ...].cpu().numpy(), '6')
    # visual2d(y_train[2, ...].cpu().numpy(), '7')
    # visual2d(y_train[3, ...].cpu().numpy(), '8')
    train_loader = DataLoader(TensorDataset(x_train.contiguous(), y_train.contiguous()), batch_size=dataOpt['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test.contiguous(), y_test.contiguous()), batch_size=dataOpt['batch_size'], shuffle=False)
    train_loader_l2 = DataLoader(TensorDataset(x_train_l2.contiguous(), y_train_l2.contiguous()), batch_size=dataOpt['batch_size'], shuffle=True)
    # train_loader_l2 = DataLoader(TensorDataset(x_train[:5000,...].contiguous(), y_train[:5000,...].contiguous()), batch_size=dataOpt['batch_size'], shuffle=True)
    # train_loader = DataLoader(TensorDataset(x_train.contiguous().to(device), y_train.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=True)
    # test_loader = DataLoader(TensorDataset(x_test.contiguous().to(device), y_test.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)
    if validate:
        val_loader = DataLoader(TensorDataset(x_val.contiguous().to(device), y_val.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)

    # load the model and evaluate
    if test_mode:
        model = torch.load(dataOpt['MODEL_PATH_LOAD'])
        model.eval()
        l2loss = LpLoss(size_average=False) 
        test_l2 = 0.
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += l2loss(out, y).item()
                
     
        test_l2/= len(dataOpt['dataSize']['test'])
   
        logging.info(f" test l2 loss: {test_l2:.3e}")
        logging.info(model)      
        return test_l2
    
    ################################################################
    # training and evaluation
    ################################################################
    if model_type == 'MgNO_DC_6':
        modelOpt['normalizer'] = y_normalizer
    
    if model_type == 'MgNO_DC_6':
        # model = MgNO_DC_6(**modelOpt).to(device)
        model = torch.load('/home/liux0t/FMM/MgNO/model/MgNO_DC_6ns_merge2024-05-19 22:42:20.432201.pt')
    elif model_type == 'FNO':
        # model = FNO2d(modes1=12, modes2=12, width=80, normalizer=y_normalizer).to(device)
        model = SpectralDecoder(lift=2, if_pre_permute=False, normalizer=y_normalizer, modes=12, num_spectral_layers=5, width=64, init_scale=2, output_dim=2, padding=6, resolution=63).to(device)
    elif model_type == 'DeepONet':
        model = DeepONet_POD(branch_features=100, trunk_features=100, output_features=1, grid_size=63, V=V).to(device)
    else:
        raise NameError('invalid model_type')
    
    if log_if:    
        logging.info(count_params(model))
        logging.info(model)
    optimizer, scheduler = getOptimizerScheduler(model.parameters(), **optimizerScheduler_args)
    optimizer_newton = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    scheduler_newton = torch.optim.lr_scheduler.OneCycleLR(optimizer_newton, max_lr=1e-8,
                               div_factor=2, 
                               final_div_factor=5e1,
                               pct_start=0.1,
                               steps_per_epoch=1, 
                               epochs=optimizerScheduler_args['epochs'])
    h1loss = HsLoss(d=2, p=2, k=1, a=dataOpt['loss_weight'], size_average=False, res=y_train.size(2), relative=False)
    h1loss.cuda(device)

    h1loss = HsLoss_2(d=2, p=2, k=2, size_average=False, res=y_train.size(2), relative=False).to(device)
  
    l2loss = LpLoss(size_average=False, relative=False )   #
    # if dataOpt['loss_type']=='pde':
    pde_loss = PDEloss_GrayScott().to(device)
    l2loss_rel = LpLoss(size_average=False, relative=True )
    ############################
    def train(train_loader, optimizer, scheduler):
        model.train()
        train_l2, train_h1 = 0, 0
        # train_f_dist = torch.zeros(y_train.size(1))

        for batch_idx, (x, y) in enumerate(train_loader):
           
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if dataOpt['loss_type'] in {'l2', 'h1'}:
                out = model(x).squeeze()
            elif dataOpt['loss_type']=='pde':
                if dataOpt['GN']:
                    # grad = pde_loss.getGrad(x)
                    # x_en = x_normalizer.encode(torch.cat((x, grad), dim=1))
                    x_en = x_normalizer.encode(x)
                    out = model(x_en)
                else:
                    out = model(x)

            if dataOpt['loss_type']=='h1':
                with torch.no_grad():
                    train_l2loss = l2loss(out, y)

                train_h1loss = h1loss(out, y)
                train_h1loss.backward()
            elif dataOpt['loss_type']=='l2':
                with torch.no_grad():
                    # train_h1loss = h1loss(out.view(dataOpt['batch_size']*2,-1), y.view(dataOpt['batch_size']*2,-1))
                    train_h1loss = pde_loss(x[:,0:1,...],x[:,1:2,...], out[:,0:1,...], out[:,1:2,...])
                train_l2loss = l2loss(out, y)
                train_l2loss.backward()
            elif dataOpt['loss_type']=='pde':
                with torch.no_grad():
                    train_l2loss = l2loss(out, y)

                train_h1loss = pde_loss(x[:,0:1,...],x[:,1:2,...], out[:,0:1,...], out[:,1:2,...])
                train_h1loss.backward()              

            optimizer.step()
            train_h1 += train_h1loss.item()
            train_l2 += train_l2loss.item()
            # train_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()

        train_l2/= len(dataOpt['dataSize']['train'])
        train_h1/= len(dataOpt['dataSize']['train'])
        # train_f_dist/= len(dataOpt['dataSize']['train'])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
         
        return lr, train_l2, train_h1
            
    @torch.no_grad()
    def test(test_loader):
        model.eval()
        test_l2, test_h1 = 0., 0.
        # test_f_dist = torch.zeros(y_test.size(1))
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if dataOpt['loss_type'] in {'l2', 'h1'}:
                    out = model(x).squeeze()
                elif dataOpt['loss_type']=='pde':
                    if dataOpt['GN']:
                        # grad = pde_loss.getGrad(x)
                        # x_en = x_normalizer.encode(torch.cat((x, grad), dim=1))
                        x_en = x_normalizer.encode(x)
                        out = model(x_en)
                    else:
                        out = model(x)
                test_l2 += l2loss(out, y).item()
                test_h1loss = h1loss(out.view(out.size(0)*2,-1), y.view(out.size(0)*2,-1))
                test_h1 += test_h1loss.item()
                # test_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()
                
        test_l2/= len(dataOpt['dataSize']['test'])
        test_h1/= len(dataOpt['dataSize']['test'] )          
        # test_f_dist/= len(dataOpt['dataSize']['test'] ) 

        return  test_l2, test_h1 #, test_f_dist

        
    ############################  
    ###### start to train ######
    ############################
    
    train_h1_rec, train_l2_rec, test_l2_rec, test_h1_rec = [], [], [], []
    if validate:
        val_l2_rec, val_h1_rec = [], [],
    
    best_l2, best_test_l2, best_test_h1, arg_min_epoch = 1.0, 1.0, 1.0, 0  
    with tqdm(total=optimizerScheduler_args['epochs']*2, disable=tqdm_disable) as pbar_ep:
                            

        for epoch in range(optimizerScheduler_args['epochs']):
            desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
            
            dataOpt['loss_type'] = 'pde'
            lr, train_l2, train_h1 = train(train_loader, optimizer_newton, scheduler_newton)
            dataOpt['loss_type'] = 'l2'
            lr, train_l2, train_h1 = train(train_loader_l2, optimizer, scheduler)
            test_l2, test_h1 = test(test_loader)
            if validate:
                val_l2, val_h1 = test(val_loader)
            
            train_l2_rec.append(train_l2); train_h1_rec.append(train_h1); 
            test_l2_rec.append(test_l2); test_h1_rec.append(test_h1); 
            if validate:
                val_l2_rec.append(val_l2); val_h1_rec.append(val_h1)
            if validate:
                if val_l2 < best_l2:
                    best_l2 = val_l2
                    arg_min_epoch = epoch
                    best_test_l2 = test_l2
                    best_test_h1 = test_h1
           
            desc += f" | current lr: {lr:.3e}"
            desc += f"| train l2 loss: {train_l2:.3e} "
            desc += f"| train h1 loss: {train_h1:.3e} "
            desc += f"| test l2 loss: {test_l2:.3e} "
            desc += f"| test h1 loss: {test_h1:.3e} "
            if validate:
                desc += f"| val l2 loss: {val_l2:.3e} "
                desc += f"| val h1 loss: {val_h1:.3e} "
           
            pbar_ep.set_description(desc)
            pbar_ep.update()
            if log_if:
                logging.info(desc) 
    # plot the l2 convergence history
    if log_if:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(train_l2_rec, label='train')
        ax[0].plot(test_l2_rec, label='test')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('l2 loss')
        ax[0].legend()
        ax[0].grid(True)
        
        ax[1].plot(train_h1_rec, label='train')
        ax[1].plot(test_h1_rec, label='test')
        ax[1].set_yscale('log')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('h1 loss')
        ax[1].legend()
        ax[1].grid(True)
        
        plt.savefig(MODEL_PATH+'.png')
        plt.close()
   
    
    if model_save:
        torch.save(model, MODEL_PATH_PARA)
            
    return test_l2






if __name__ == "__main__":

    import newton_multi_solution
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data", type=str, default="ns_merge", help="data name, darcy, darcy20c6, darcy15c10, darcyF, darcy_contin")
    parser.add_argument(
            "--model_type", type=str, default="FNO", help="model type")
    parser.add_argument(
            "--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument(
            "--batch_size", type=int, default=50, help="batch size")
    parser.add_argument(
            "--optimizer_type", type=str, default="adam", help="optimizer type")
    parser.add_argument(
            "--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
            "--final_div_factor", type=float, default=10, help="final_div_factor")
    parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
            "--loss_type", type=str, default="l2", help="loss type, l2, h1, pde")
    parser.add_argument(
            "--GN", action='store_true', help="use normalized x")
    parser.add_argument(
            "--sample_x", action='store_true', help="sample x")
    parser.add_argument(
            "--sampling_rate", type=int, default=1, help="sampling rate")
    parser.add_argument(
            "--normalizer", action='store_true', help="use normalizer")
    parser.add_argument(
            "--normalizer_type", type=str, default="GN", help="PGN, GN")
    parser.add_argument(
            "--num_layer", type=int, default=5, help="number of layers")
    parser.add_argument(
            "--num_channel_u", type=int, default=24, help="number of channels for u")
    parser.add_argument(
            "--num_channel_f", type=int, default=1, help="number of channels for f")
    parser.add_argument(
            '--num_iteration', type=list, nargs='+', default=[[1,0], [1,0], [1,0], [1,1], [2,0]], help='number of iterations in each layer')
    parser.add_argument(
            '--padding_mode', type=str, default='zeros', help='padding mode')
    parser.add_argument(
            '--last_layer', type=str, default='linear', help='last layer type')

    parser.add_argument(
            "--test", action='store_true', help="load model and test")
    parser.add_argument(
            "--MODEL_PATH_LOAD", type=str, default="PATH_LOAD", help="PATH_LOAD")
    args = parser.parse_args()
    args = vars(args)

    for i in range(len(args['num_iteration'])):
        for j in range(len(args['num_iteration'][i])):
            args['num_iteration'][i][j] = int(args['num_iteration'][i][j])
        

    if  args['sample_x']:
        if args['data'] in {'darcy', 'darcy20c6', 'darcy15c10', 'darcyF', 'darcy_contin'}:
            args['sampling_rate'] = 2
        elif args['data'] == 'a4f1':
            args['sampling_rate'] = 4
        elif args['data'] == 'helm':
            args['sampling_rate'] = 1
        elif args['data'] == 'pipe':
            args['sampling_rate'] = 1


    
  
        
    dataOpt = {}
    dataOpt['data'] = args['data']
    dataOpt['sampling_rate'] = args['sampling_rate']
    dataOpt['sample_x'] = args['sample_x']
    dataOpt['batch_size'] = args['batch_size']
    dataOpt['loss_type']=args['loss_type']
    dataOpt['loss_weight'] = [2,]
    dataOpt['normalizer_type'] = args['normalizer_type']
    dataOpt['GN'] = args['GN']
    dataOpt['MODEL_PATH_LOAD'] = args['MODEL_PATH_LOAD']
    dataOpt = getDataSize(dataOpt)

    modelOpt = {}
    modelOpt['num_layer'] = args['num_layer']
    modelOpt['num_channel_u'] = args['num_channel_u']
    modelOpt['num_channel_f'] = args['num_channel_f']
    modelOpt['num_classes'] = 1
    modelOpt['num_iteration'] = args['num_iteration']
    modelOpt['in_chans'] = 1
    modelOpt['normalizer'] = args['normalizer'] 
    modelOpt['output_dim'] = 1
    modelOpt['activation'] = 'gelu'    
    modelOpt['padding_mode'] = args['padding_mode']
    modelOpt['last_layer'] = args['last_layer']

    optimizerScheduler_args = {}
    optimizerScheduler_args['optimizer_type'] = args['optimizer_type']
    optimizerScheduler_args['lr'] = args['lr']
    optimizerScheduler_args['weight_decay'] = args['weight_decay']
    optimizerScheduler_args['epochs'] = args['epochs']
    optimizerScheduler_args['final_div_factor'] = args['final_div_factor']
    optimizerScheduler_args['div_factor'] = 2

    newton_multi_solution.objective(dataOpt, modelOpt, optimizerScheduler_args, model_type=args['model_type'],
    validate=False, tqdm_disable=True, log_if=True, 
    model_save=True, test_mode=args['test'])





    

    