import torch
import torch.nn as nn
import torch.nn.functional as F

import os, logging
import numpy as np
import matplotlib.pyplot as plt

from utilities3 import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import argparse
torch.set_printoptions(threshold=500000)


class Conv_Dyn(nn.Module):
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=1, bias=False, padding_mode='replicate', resolution=480):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0, bias=True, padding_mode=padding_mode)

    def forward(self, out):
        u, f, a, diva, r = out
        if diva is None:
            diva = self.conv_1(a)
        if u is None:
            r = f if r is None else r 
            u = self.conv_0(torch.tanh(diva)) * r
        else:
            r = f - diva * u
            u = u + self.conv_0(torch.tanh(diva)) * r                             
        out = (u, f, a, diva, r)
        return out

class MgRestriction(nn.Module): 
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=2, padding=0, bias=False, padding_mode='zeros'):
        super().__init__()

        self.R_1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0, bias=bias, padding_mode=padding_mode)
        self.R_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.R_3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)

    def forward(self, out):
        u_old, f_old, a_old, diva, r_old = out
        if diva is None:
            a = self.R_1(a_old)  
        else:
            a = a_old                          
        r = self.R_3(r_old)                               
        out = (None, None, a, diva, r)
        return out
    

class MG_FEM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_iterations=[1, 1, 1, 1, 1]):
        super().__init__()
        self.num_iterations = num_iterations
        self.resolutions = [127, 63, 31, 15, 7, 4]
        
        self.transpose_layers = nn.ModuleList()
        for _ in range(len(num_iterations) - 1):
            self.transpose_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=2, 
                    padding=0, 
                    bias=False
                )
            )

        self.conv_layers = nn.ModuleList()
        layers = []
        for layer_index, num_iterations_layer in enumerate(num_iterations):
            for _ in range(num_iterations_layer):
                layers.append(Conv_Dyn(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    resolution=self.resolutions[layer_index]
                ))
            self.conv_layers.append(nn.Sequential(*layers))

            if layer_index < len(num_iterations) - 1:
                layers = [MgRestriction(in_channels=in_channels, out_channels=out_channels)]

    def forward(self, u, f, a, diva_list=[None for _ in range(7)], r=None):
   
        out_list = [None] * len(self.num_iterations)
        
        for layer_index in range(len(self.num_iterations)):
            out = (u, f, a, diva_list[layer_index], r)
            u, f, a, diva, r = self.conv_layers[layer_index](out)
            out_list[layer_index] = (u, f, a, r)
            if diva_list[layer_index] is None:
                diva_list[layer_index] = diva

        for layer_index in range(len(self.num_iterations) - 2, -1, -1):
            u, f, a, r = out_list[layer_index]
            u_post = u + self.transpose_layers[layer_index](out_list[layer_index + 1][0])
            out_list[layer_index] = (u_post, f, a, r)
            
        return out_list[0][0], out_list[0][1], out_list[0][2], diva_list

class MgNO(nn.Module):
    def __init__(self,
                    dim_input = 4,  
                    features = 12,
                    layers = 4,
                    ):
        super(MgNO, self).__init__()

        mean_sos, std_sos =  torch.tensor(1488.3911, dtype = torch.float32), torch.tensor(27.5279, dtype = torch.float32)

        self.register_buffer("mean_sos", mean_sos)
        self.register_buffer("std_sos", std_sos)
        self.layers = layers
        self.lifting_1 = nn.Conv2d(2, features, kernel_size=1, bias=False)
        self.lifting_2 = nn.Conv2d(1, features, kernel_size=1, bias=False)
        self.lifting_3 = nn.Conv2d(2, features, kernel_size=1)

        self.proj = nn.Conv2d(features, 1, kernel_size=1)
     
        self.mgno = nn.ModuleList()
        for l in range(2):
            self.mgno.append(MG_FEM(in_channels=features, out_channels=features, num_iteration=[1,1,1,1,1,1,]))

    def forward(self, f, a):
        # theta is 100,480,480,2
        # normalize
        # sos = (sos - self.mean_sos) / (self.std_sos * .6)
      
        u = None
        r = None
        f = self.lifting_2(f) 
        a = self.lifting_3(a)
 
        u,f,a,diva_list = self.mgno[0](u,f,a,r=r,diva_list=[None for _ in range(6)])# batch,feature,x,y:   100,feature_,960+pad,960+pad
        u_list = []
        u,f,a,diva_list = self.mgno[1](u,f,a,r=r,diva_list=[None for _ in range(6)])
        # u_list.append(self.proj(u.clone()))
        for _ in range(5):
            u,f,a,diva_list = self.mgno[1](u,f,a,diva_list)  
            u_list.append(self.proj(u.clone()))
        return u_list


def objective(dataOpt, modelOpt, optimizerScheduler_args,
                tqdm_disable=True, 
              log_if=False, validate=False, model_type='MgNO', 
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
   
    data = torch.load('/home/liux0t/neural_MG/pytorch/darcy_ufa_4.pt')
    u = data['u_0'].detach()
    resolution = u.size(2)
    f = data['y_train_0'].detach()
    a = data['x_train_0'].detach()
    avg_a_kernl = 1./3 * torch.tensor([[[[1.,1],[1,0,]]], [[[0.,1],[1,1,]]]],).to(device)
    a = torch.nn.functional.conv2d(a, avg_a_kernl)

    train_loader = DataLoader(TensorDataset(u[0:2000], f[0:2000], a[0:2000]), batch_size=10, shuffle=True)
    test_loader = DataLoader(TensorDataset(u[000:2200], f[000:2200], a[000:2200]), batch_size=10, shuffle=False)
   
    ################################################################
    # training and evaluation
    ################################################################
    
    # model = MgNO(dim_input=4, features=32, ).to(device)
    model = torch.load('/home/liux0t/FMM/MgNO/model/MgNO_DCdarcy_ufa_42024-07-23 12:19:02.705444.pt')    
    if log_if:    
        logging.info(count_params(model))
        logging.info(model)
    optimizer, scheduler = getOptimizerScheduler(model.parameters(), **optimizerScheduler_args)
    
    h1loss = HsLoss(d=2, p=2, k=1, a=dataOpt['loss_weight'], size_average=False, res=resolution,)
    h1loss.cuda(device)
    h1loss = PDEloss().to(device)

    l2loss = LpLoss(size_average=False)  
    ############################
    def train(train_loader):
        model.train()
        train_l2, train_h1 = 0, 0
        train_f_dist = torch.zeros(resolution)

        for u, f, a in train_loader:
            u = u.to(device)
            f = f.to(device)
            a = a.to(device)    
            optimizer.zero_grad()
            out = model(f, a)
            if dataOpt['loss_type']=='h1':
                with torch.no_grad():
                    train_l2loss = l2loss(out, u)

                train_h1loss = h1loss(out, f, a)
                train_h1loss.backward()
            else:
                with torch.no_grad():
                    train_h1loss = h1loss(out[-1], f, a)
                
                # train_l2loss = l2loss(out[-1], u)
                train_l2loss = sum([2.1**i*l2loss(out_, u) for i, out_ in enumerate(out)])
                train_l2loss.backward()

            optimizer.step()
            train_h1 += train_h1loss.item()
            train_l2 += train_l2loss.item()

        train_l2/= len(dataOpt['dataSize']['train'])
        train_h1/= len(dataOpt['dataSize']['train'])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
         
        return lr, train_l2, train_h1
            
    @torch.no_grad()
    def test(test_loader):
        model.eval()
        test_l2, test_h1 = 0., 0.
        test_f_dist = torch.zeros(resolution)
        with torch.no_grad():
            for u, f, a in test_loader:
                u = u.to(device)
                f = f.to(device)
                a = a.to(device)
                out = model(f, a)
                test_l2 += l2loss(out[-1], u).item()
                test_h1loss = h1loss(out[-1], f, a)
                test_h1 += test_h1loss.item()
                
        test_l2/= len(dataOpt['dataSize']['test'])
        test_h1/= len(dataOpt['dataSize']['test'] )          

        return  test_l2, test_h1

    @torch.no_grad()
    def test_iter(test_loader):
        model.eval()
        test_l2_1, test_l2_2, test_l2_3 = 0., 0., 0.
        test_f_dist = torch.zeros(resolution)
        with torch.no_grad():
            for u, f, a in test_loader:
                u = u.to(device)
                f = f.to(device)
                a = a.to(device)
                out = model(f, a)
                test_l2_1 += l2loss(out[-1], u).item()
                test_l2_2 += l2loss(out[-2], u).item()
                test_l2_3 += l2loss(out[-3], u).item()

        test_l2_1/= len(test_loader)*10
        test_l2_2/= len(test_loader)*10      
        test_l2_3/= len(test_loader)*10
        return  test_l2_1, test_l2_2, test_l2_3

        
    ############################  
    ###### start to train ######
    ############################
    
    if test_mode == True:
        test_l2_1, test_l2_2, test_l2_3 = test_iter(test_loader)
        print(f"test l2 loss: {test_l2_1:.3e}, {test_l2_2:.3e}, {test_l2_3:.3e}")
        return 

    train_h1_rec, train_l2_rec, train_f_dist_rec, test_l2_rec, test_h1_rec, test_f_dist_rec = [], [], [], [], [], []
    if validate:
        val_l2_rec, val_h1_rec = [], [],
    
    best_l2, best_test_l2, best_test_h1, arg_min_epoch = 1.0, 1.0, 1.0, 0  
    with tqdm(total=optimizerScheduler_args['epochs'], disable=tqdm_disable) as pbar_ep:
                            
        for epoch in range(optimizerScheduler_args['epochs']):
            desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
            lr, train_l2, train_h1  = train(train_loader)
            test_l2, test_h1  = test(test_loader)
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


        if log_if and validate: 
            logging.info(f" test h1 loss: {best_test_h1:.3e}, test l2 loss: {best_test_l2:.3e}")
                 
        # if log_if:
        #     logging.info('train l2 rec:')
        #     logging.info(train_l2_rec)
        #     logging.info('train h1 rec:')
        #     logging.info(train_h1_rec)
        #     logging.info('test l2 rec:')
        #     logging.info(test_l2_rec)
        #     logging.info('test h1 rec:')
        #     logging.info(test_h1_rec)
        #     logging.info('train_f_dist_rec')
        #     logging.info(torch.stack(train_f_dist_rec))
        #     logging.info('test_f_dist_rec')
        #     logging.info(torch.stack(test_f_dist_rec))
 
            
    if model_save:
        torch.save(model, MODEL_PATH_PARA)
            
    return test_l2






if __name__ == "__main__":

    import darcy_ufa_2
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data", type=str, default="darcy_ufa_4", help="data name, darcy, darcy20c6, darcy15c10, darcyF, darcy_contin")
    parser.add_argument(
            "--model_type", type=str, default="MgNO_DC", help="FNO, MgNO")
    parser.add_argument(
            "--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument(
            "--batch_size", type=int, default=10, help="batch size")
    parser.add_argument(
            "--optimizer_type", type=str, default="adamw", help="optimizer type")
    parser.add_argument(
            "--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument(
            "--final_div_factor", type=float, default=40, help="final_div_factor")
    parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
            "--loss_type", type=str, default="l2", help="loss type, l2, h1")
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
            "--num_channel_f", type=int, default=5, help="number of channels for f")
    parser.add_argument(
            '--num_iteration', type=list, nargs='+', default=[[1,0], [1,0], [1,0], [2,0], [2,0]], help='number of iterations in each layer')
    parser.add_argument(
            '--padding_mode', type=str, default='reflect', help='padding mode')
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
    dataOpt['dataSize'] = {'train': range(2000), 'test': range(2000, 2200)}

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
    optimizerScheduler_args['div_factor'] = 1

    darcy_ufa_2.objective(dataOpt, modelOpt, optimizerScheduler_args, model_type=args['model_type'],
    validate=False, tqdm_disable=True, log_if=False, 
    model_save=True, test_mode=args['test'])





    

    