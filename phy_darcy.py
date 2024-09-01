import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MgNO_DC, MgNO_DC_smooth, MgNO_NS, MgNO_helm, MgNO_DC_2, MgNO_DC_3, MgNO_DC_4, MgNO_DC_5, MgNO_DC_31
from models3 import HANO, HANO_DC
# from get_Unet_new import get_model
import os, logging
import numpy as np
import matplotlib.pyplot as plt

from utilities3 import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import argparse
torch.set_printoptions(threshold=500000)


class ConvSlice(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super().__init__()
    
    # initiate the conv2d with the weight of gradx and grady
        self.avg_a_kernl = 1./3 * torch.tensor([[[[1.,1],[1,0,]]], [[[0.,1],[1,1,]]]],).to('cuda')
        k00 = .5 * torch.tensor([[[[0.,1],[1, 2,]], [[2.,1],[1,0,]]]],)
        k01 = .5 * torch.tensor([[[[0.,1],[0, 0,]], [[1.,0],[0,0,]]]],)
        k_10 = .5 * torch.tensor([[[[0.,0],[1, 0,]], [[1.,0],[0,0,]]]],)
        k10 = .5 * torch.tensor([[[[0.,0],[0, 1,]], [[0.,1],[0,0,]]]],)
        k0_1 = .5 * torch.tensor([[[[0.,0],[0, 1,]], [[0.,0],[1,0,]]]],)
        # concate the kernels to be the shape of [5,1,3,3]
        kernel = torch.cat([k00, k01, k10, k_10, k0_1], dim=0)
        # initialize the weight of the conv1 to be kernel
        self.conv1 = nn.Conv2d(2, 5, kernel_size=2, stride=1, padding=0, padding_mode=padding_mode, bias=False)
        self.conv1.weight = nn.Parameter(kernel)
        # set the weight to be untrainable
        self.conv1.weight.requires_grad = False
    def forward(self, coef):
        # Extract the sliding local blocks from the image tensor
        coef_avg = F.conv2d(coef, self.avg_a_kernl, stride=1, padding=0)
        result = self.conv1(coef_avg)
        return result

class PDEloss_2(nn.MSELoss):
    def __init__(self, reduction='sum', res=256):
        super().__init__(reduction=reduction)
        kernel00 = torch.tensor([[[[0.,0,0],[0,1,0],[0,0,0]]]])
        kernel01 = torch.tensor([[[[0.,-1,0],[0,0,0],[0,0,0]]]])
        kernel10 = torch.tensor([[[[0.,0,0],[0,0,-1],[0,0,0]]]])
        kernel0_1 = torch.tensor([[[[0.,0,0],[0,0,0],[0,-1,0]]]])
        kernel_10 = torch.tensor([[[[0.,0,0],[-1,0,0],[0,0,0]]]])
        # concate the kernels to be the shape of [5,1,3,3]
        kernel = torch.cat([kernel00, kernel01, kernel10, kernel_10, kernel0_1, ], dim=0)
        # initialize the weight of the conv1 to be kernel

        self.register_buffer('kernel', kernel)
        # register the kernels as a buffer
  
        f = torch.ones(res, res).reshape(1, res, res)/ (res**2)
        self.register_buffer('f', f)
    
    def diva_fem(self, u, a):
        # u is in shape of batch*1*res*res, a is in shape of batch*5*res*res
        # F.conv2d(u, self.kernel, padding=1) is in shape of batch*5*res*res
        # return the F.conv2d(u, self.kernel, padding=1) * a and sum over the channel dimension
  
        return torch.sum(F.conv2d(u, self.kernel, padding=1) * a, dim=1, keepdim=True)
    
    def forward(self, u, a, f=None):
        if f is None:
            f = self.f
        # loss = torch.linalg.norm(self.diva_fem(u, a)-f)
        # loss = super().forward(self.diva_fem(u, a), f)
        dudu = self.diva_fem(u, a) * u
        fu = f * u
        loss = torch.sum(1/2 * dudu - fu)  
        return loss 


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
   
    data = torch.load('/home/liux0t/neural_MG/pytorch/darcy20c6_fem.pt')
    y = data['u'].to(device)
    x = data['a'].unsqueeze(1).to(device)
    convslice = ConvSlice().to(device)

    x = convslice(x)
    dataOpt['dataSize'] = {'train': range(1280), 'test': range(1280, 1280+110), 'val':range(1280+110+110)}

    x_train = x[dataOpt['dataSize']['train'],...]
    y_train = y[dataOpt['dataSize']['train'],...]
    x_test = x[dataOpt['dataSize']['test'],...]
    y_test = y[dataOpt['dataSize']['test'],...]
    x_val = x[dataOpt['dataSize']['val'],...]
    y_val = y[dataOpt['dataSize']['val'],...]
    x_train = x_train.detach()
    y_train = y_train.detach()
    x_test = x_test.detach()
    y_test = y_test.detach()
    x_val = x_val.detach()
    y_val = y_val.detach()

    y_normalizer = GaussianNormalizer(y_train)
   

    f = torch.ones(size=(1,1,255,255)).to(device)
    f = f.to(device)/255**2
    criterion_2 = PDEloss_2().to(device)

    if modelOpt['normalizer']:
        modelOpt['normalizer'] = y_normalizer

    if x_train.ndim == 3:
        x_train = x_train[:, np.newaxis, ...]
        x_test = x_test[:, np.newaxis, ...]
        if validate:
            x_val = x_val[:, np.newaxis, ...]
    train_loader = DataLoader(TensorDataset(x_train.contiguous(), y_train.contiguous()), batch_size=dataOpt['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test.contiguous(), y_test.contiguous()), batch_size=dataOpt['batch_size'], shuffle=False)
    
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
    
    model = MgNO_DC_5(**modelOpt).to(device)
    # model = torch.load('/home/liux0t/FMM/MgNO/model/MgNO_DCdarcy20c62023-12-26 10:29:03.589766.pt')    
    if log_if:    
        logging.info(count_params(model))
        logging.info(model)
    optimizer, scheduler = getOptimizerScheduler(model.parameters(), **optimizerScheduler_args)
    
    h1loss = HsLoss(d=2, p=2, k=1, a=dataOpt['loss_weight'], size_average=False, res=y_train.size(2),)
    h1loss.cuda(device)
    if dataOpt['data'] == 'helm':
        h1loss = HSloss_d()
    l2loss = LpLoss(size_average=False)  
    ############################
    def train(train_loader):
        model.train()
        train_l2, train_h1 = 0, 0
        train_f_dist = torch.zeros(y_train.size(2))

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if dataOpt['loss_type']=='h1':
                with torch.no_grad():
                    train_l2loss = l2loss(out, y)

                # train_h1loss, train_f_l2loss, f_l2x, f_l2y = h1loss(out, y)
                _, _, f_l2x, f_l2y = h1loss(out, y)
                train_h1loss = criterion_2(out, x, f)
                train_h1loss.backward()
            else:
                with torch.no_grad():
                    # train_h1loss, train_f_l2loss, f_l2x, f_l2y = h1loss(out, y)
                    train_h1loss = criterion_2(out, x, f)

                train_l2loss = l2loss(out, y)
                train_l2loss.backward()

            optimizer.step()
            train_h1 += train_h1loss.item()
            train_l2 += train_l2loss.item()
            train_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()

        train_l2/= len(dataOpt['dataSize']['train'])
        train_h1/= len(dataOpt['dataSize']['train'])
        train_f_dist/= len(dataOpt['dataSize']['train'])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
         
        return lr, train_l2, train_h1, train_f_dist
            
    @torch.no_grad()
    def test(test_loader):
        model.eval()
        test_l2, test_h1 = 0., 0.
        test_f_dist = torch.zeros(y_test.size(2))
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += l2loss(out, y).item()
                _, _, f_l2x, f_l2y = h1loss(out, y)
                test_h1loss = criterion_2(out, x, f)
                test_h1 += test_h1loss.item()
                test_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()
                
        test_l2/= len(dataOpt['dataSize']['test'])
        test_h1/= len(dataOpt['dataSize']['test'] )          
        test_f_dist/= len(dataOpt['dataSize']['test'] ) 

        return  test_l2, test_h1, test_f_dist

        
    ############################  
    ###### start to train ######
    ############################
    
    train_h1_rec, train_l2_rec, train_f_dist_rec, test_l2_rec, test_h1_rec, test_f_dist_rec = [], [], [], [], [], []
    if validate:
        val_l2_rec, val_h1_rec = [], [],
    
    best_l2, best_test_l2, best_test_h1, arg_min_epoch = 1.0, 1.0, 1.0, 0  
    with tqdm(total=optimizerScheduler_args['epochs'], disable=tqdm_disable) as pbar_ep:
                            
        for epoch in range(optimizerScheduler_args['epochs']):
            desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
            lr, train_l2, train_h1, train_f_dist = train(train_loader)
            test_l2, test_h1, test_f_dist = test(test_loader)
            if validate:
                val_l2, val_h1 = test(val_loader)
            
            train_l2_rec.append(train_l2); train_h1_rec.append(train_h1); train_f_dist_rec.append(train_f_dist)
            test_l2_rec.append(test_l2); test_h1_rec.append(test_h1); test_f_dist_rec.append(test_f_dist)
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
                 
        if log_if:
            logging.info('train l2 rec:')
            logging.info(train_l2_rec)
            logging.info('train h1 rec:')
            logging.info(train_h1_rec)
            logging.info('test l2 rec:')
            logging.info(test_l2_rec)
            logging.info('test h1 rec:')
            logging.info(test_h1_rec)
            logging.info('train_f_dist_rec')
            logging.info(torch.stack(train_f_dist_rec))
            logging.info('test_f_dist_rec')
            logging.info(torch.stack(test_f_dist_rec))
 
            
    if model_save:
        torch.save(model, MODEL_PATH_PARA)
            
    return test_l2






if __name__ == "__main__":

    import phy_darcy
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data", type=str, default="ns_merge", help="data name, darcy, darcy20c6, darcy15c10, darcyF, darcy_contin")
    parser.add_argument(
            "--model_type", type=str, default="MgNO_DC", help="FNO, MgNO")
    parser.add_argument(
            "--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument(
            "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
            "--optimizer_type", type=str, default="adam", help="optimizer type")
    parser.add_argument(
            "--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
            "--final_div_factor", type=float, default=100, help="final_div_factor")
    parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
            "--loss_type", type=str, default="h1", help="loss type, l2, h1")
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

    phy_darcy.objective(dataOpt, modelOpt, optimizerScheduler_args, model_type=args['model_type'],
    validate=False, tqdm_disable=True, log_if=True, 
    model_save=True, test_mode=args['test'])





    

    