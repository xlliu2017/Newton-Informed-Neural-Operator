import torch
from utilities3 import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import navier
from navier import test
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "--data", type=str, default="1e-5", help="data name, darcy, darcy20c6, darcy15c10, darcyF, darcy_contin"
)
parser.add_argument(
        "--model_type", type=str, default="MgNO_DC_3", help="UNO, FNO, MgNO_NS"
)
parser.add_argument(
        "--epochs", type=int, default=500, help="number of epochs")
parser.add_argument(
        "--batch_size", type=int, default=50, help="batch size"
)
parser.add_argument(
        "--lr", type=float, default=6e-4, help="learning rate")
parser.add_argument(
        "--final_div_factor", type=float, default=100, help="final_div_factor")
parser.add_argument(
        "--div_factor", type=float, default=2, help="div_factor")
parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="weight decay")
parser.add_argument(
        "--loss_type", type=str, default="l2", help="loss type, l2, h1")

parser.add_argument(
        "--num_layer", type=int, default=5, help="number of layers"
)
parser.add_argument(
        "--num_channel_u", type=int, default=32, help="number of channels in u"
)
parser.add_argument(
        "--num_channel_f", type=int, default=1, help="number of channels in f"
)
parser.add_argument(
        '--num_iteration', type=list, nargs='+', default=[[1,0], [1,0], [1,0], [2,0], [2,0]],  help='number of iterations in each layer')
parser.add_argument('--padding_mode', type=str, default='circular', help='padding mode')
parser.add_argument('--mlp_hidden_dim', type=int, default=0)
parser.add_argument('--bias', action='store_true',)
args = parser.parse_args([])
args = vars(args)

for i in range(len(args['num_iteration'])):
    for j in range(len(args['num_iteration'][i])):
        args['num_iteration'][i][j] = int(args['num_iteration'][i][j])


dataOpt = {}
dataOpt['data'] = args['data']
dataOpt['path'] =  getPath(args['data'], flag=None)#'/ibex/ai/home/liux0t/Xinliang/FMM/data/ns_V1e-3_N5000_T50.mat' #'/ibex/ai/home/liux0t/Xinliang/FMM/data/ns_V1e-4_N10000_T30.mat'#'/ibex/ai/home/liux0t/Xinliang/FMM/data/NavierStokes_V1e-5_N1200_T20.mat' ##
dataOpt['ntrain'] = 1000
dataOpt['ntest'] = 100
dataOpt['batch_size'] = 50
dataOpt['epochs'] = 500
dataOpt['T_in'] = 1
dataOpt['T_out'] = 1
dataOpt['T'] = 19
dataOpt['step'] = 1
dataOpt['r'] = 64
dataOpt['sampling'] = 1
dataOpt['full_train'] = True
dataOpt['full_train_2'] = True
dataOpt['loss_type'] = args['loss_type']
dataOpt['GN'] = False
dataOpt['learning_rate'] = args['lr']
dataOpt['final_div_factor'] = args['final_div_factor']
dataOpt['div_factor'] = args['div_factor']
dataOpt['weight_decay'] = args['weight_decay']

train_a, train_u, test_a, test_u = getNavierDataSet3(dataOpt, device, return_normalizer=None, GN=None, normalizer=None)
# train_a = train_a.permute(0, 3, 1, 2).contiguous().to(device)
# train_u = train_u.permute(0, 3, 1, 2).contiguous().to(device)
test_a = test_a.permute(0, 3, 1, 2).contiguous().to(device)
test_u = test_u.permute(0, 3, 1, 2).contiguous().to(device)

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=dataOpt['batch_size']*10, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=dataOpt['batch_size'], shuffle=False)

model = torch.load('/home/liux0t/FMM/MgNO/model/HANO1e-52023-12-08 23:34:42.003738.pt')
l2loss = LpLoss(size_average=False)   
train_l2_step, train_l2_full, test_l2_step, test_l2_full, test_l2_full_2 = 0, 0, 0, 0, 0  
test_l2_full, test_l2_full_2, test_l2_step = test(model, args['model_type'], l2loss, test_loader, test_l2_full, test_l2_full_2, test_l2_step, dataOpt)
test_l2_full / dataOpt['ntest']

trainLossFunc = LpLoss(size_average=False)
train_l2_step, train_l2_full, test_l2_step, test_l2_full, test_l2_full_2 = 0, 0, 0, 0, 0  
test_l2_full, test_l2_full_2, test_l2_step = test(model, args['model_type'], l2loss, test_loader, test_l2_full, test_l2_full_2, test_l2_step, dataOpt)
print(test_l2_full / dataOpt['ntest'])
xx, yy = next(iter(test_loader))
loss = 0
pred_list = []
for t in range(0, dataOpt['T'], dataOpt['step']):
    y = yy[:, t:t + dataOpt['step'], ...]
    im = model(xx)
    loss += trainLossFunc(im, y)
    
    if t == 0:
        pred = im
        loss1 = loss.clone().detach()
    else:
        pred = torch.cat((pred, im), dim=1)

    xx = torch.cat((xx[:, dataOpt['step']:, ...], im), dim=1)


#  plot the sequency of pred
from plotly.subplots import make_subplots
import plotly.graph_objects as go
i = 0
COLORSCALE = 'RdYlBu'
data0=go.Contour(z=yy[i, 0, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)

data1 = go.Contour(z=yy[i, 3, ...].squeeze().detach().cpu(),
colorscale=COLORSCALE,
# zmax=3,
# zmin=-3,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data11 = go.Contour(z=pred[i, 3, ...].squeeze().detach().cpu(),
colorscale=COLORSCALE,
# zmax=3,
# zmin=-3,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data12 = go.Contour(z=(yy[i, 3, ...]-pred[i, 3, ...]).squeeze().detach().cpu(),
colorscale=COLORSCALE,
zmax=0.15,
zmin=-0.15,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)



data2=go.Contour(z=yy[i, 6, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
   coloring='heatmap',),
)
data21=go.Contour(z=pred[i, 6, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
   coloring='heatmap',),
)

data22=go.Contour(z=(yy[i, 6, ...]-pred[i, 6, ...]).squeeze().detach().cpu(),
colorscale=COLORSCALE,
zmax=0.15,
zmin=-0.15,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)


data3=go.Contour(z=yy[i, 9, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
   coloring='heatmap',),
)
data31=go.Contour(z=pred[i, 9, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
   coloring='heatmap',),
)
data32=go.Contour(z=(yy[i, 9, ...]-pred[i, 9, ...]).squeeze().detach().cpu(),
colorscale=COLORSCALE,
zmax=0.15,
zmin=-0.15,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)

data4=go.Contour(z=yy[i, 12, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data41=go.Contour(z=pred[i, 12, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data42=go.Contour(z=(yy[i, 12, ...]-pred[i, 12, ...]).squeeze().detach().cpu(),
colorscale=COLORSCALE,
zmax=0.15,
zmin=-0.15,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)


data5=go.Contour(z=yy[i, 15, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data51=go.Contour(z=pred[i, 15, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data52=go.Contour(z=(yy[i, 15, ...]-pred[i, 15, ...]).squeeze().detach().cpu(),
colorscale=COLORSCALE,
zmax=0.15,
zmin=-0.15,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),)

data6=go.Contour(z=yy[i, 18, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data61=go.Contour(z=pred[i, 18, ...].detach().cpu(),
colorscale=COLORSCALE,
# zmax=0.05,
# zmin=-0.05,
line_smoothing=0.85,
line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data62=go.Contour(z=(yy[i, 18, ...]-pred[i, 18, ...]).squeeze().detach().cpu(),
colorscale=COLORSCALE,
zmax=0.15,
zmin=-0.15,
line_smoothing=0.85,
line_width=0.1,

contours=dict(
    coloring='heatmap',),
)

# plot the third row for the error yy-pred  


colorbar_trace  = go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             marker=dict(
                                 colorscale='RdYlBu', 
                                 showscale=True,
                                 cmin=-5,
                                 cmax=5,
                                 colorbar=dict(thickness=5, tickvals=[-5, 5], ticktext=['Low', 'High'], outlinewidth=0)
                             ),
                             hoverinfo='none'
                            )

fig = make_subplots(rows=2, cols=7, subplot_titles=("t=0","t=3", "t=6", "t=9", "t=12", "t=15", "t=19"), horizontal_spacing=0.001, vertical_spacing = 0.003) #
# add title for each subplot
# fig.add_trace(data_a, row=1, col=1,)
fig.add_trace(data0, row=1, col=1,)
fig.add_trace(data1, row=1, col=2,)
fig.add_trace(data11, row=2, col=2,)

# fig.add_trace(colorbar_trace)
fig.add_trace(data2, row=1, col=3)
fig.add_trace(data21, row=2, col=3)
fig.add_trace(data3, row=1, col=4)
fig.add_trace(data31, row=2, col=4)
fig.add_trace(data4, row=1, col=5)
fig.add_trace(data41, row=2, col=5)
fig.add_trace(data5, row=1, col=6)
fig.add_trace(data51, row=2, col=6)
fig.add_trace(data6, row=1, col=7)
fig.add_trace(data61, row=2, col=7)


# fig = make_subplots(rows=1, cols=1)
# fig.add_trace(data1, row=1, col=1,)
fig.update_traces(showscale=False, row=1, col=1)
fig.update_traces(showscale=False, row=1, col=2)
fig.update_traces(showscale=False, row=1, col=3)
fig.update_traces(showscale=False, row=1, col=4)
fig.update_traces(showscale=False, row=1, col=5)
fig.update_traces(showscale=False, row=1, col=6)
fig.update_traces(showscale=False, row=2, col=1)
fig.update_traces(showscale=False, row=2, col=2)
fig.update_traces(showscale=False, row=2, col=3)
fig.update_traces(showscale=False, row=2, col=4)
fig.update_traces(showscale=False, row=2, col=5)
fig.update_traces(showscale=False, row=2, col=6)



fig.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                          width=1500, height=300, showlegend=False,
                    xaxis=dict(showticklabels=False),
                    xaxis2=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    yaxis2=dict(showticklabels=False),
                    xaxis3=dict(showticklabels=False),
                    xaxis4=dict(showticklabels=False),
                    yaxis3=dict(showticklabels=False),
                    yaxis4=dict(showticklabels=False), 
                    xaxis5=dict(showticklabels=False),
                    yaxis5=dict(showticklabels=False),
                    xaxis6=dict(showticklabels=False),
                    yaxis6=dict(showticklabels=False),
                    xaxis7=dict(showticklabels=False),
                    yaxis7=dict(showticklabels=False),
                    xaxis8=dict(showticklabels=False),
                    xaxis9=dict(showticklabels=False),
                    yaxis8=dict(showticklabels=False),
                    yaxis9=dict(showticklabels=False),
                    xaxis10=dict(showticklabels=False),
                    xaxis11=dict(showticklabels=False),
                    yaxis10=dict(showticklabels=False),
                    yaxis11=dict(showticklabels=False), 
                    xaxis12=dict(showticklabels=False),
                    yaxis12=dict(showticklabels=False),
                    xaxis13=dict(showticklabels=False),
                    yaxis13=dict(showticklabels=False),
                    xaxis14=dict(showticklabels=False),
                    yaxis14=dict(showticklabels=False),
                    xaxis15=dict(showticklabels=False),
                    yaxis15=dict(showticklabels=False),
                    xaxis16=dict(showticklabels=False),
                    xaxis17=dict(showticklabels=False),
                    yaxis16=dict(showticklabels=False),
                    yaxis17=dict(showticklabels=False),
                    xaxis18=dict(showticklabels=False),
                    yaxis18=dict(showticklabels=False),
                    xaxis19=dict(showticklabels=False),
                    yaxis19=dict(showticklabels=False),
                    xaxis20=dict(showticklabels=False),
                    yaxis20=dict(showticklabels=False),
                    xaxis21=dict(showticklabels=False),
                    yaxis21=dict(showticklabels=False),
                 
                 )

fig.show()

fig2 = make_subplots(rows=1, cols=7, horizontal_spacing=0.001, vertical_spacing = 0.001) #
fig2.add_trace(data12, row=1, col=2)
fig2.add_trace(data22, row=1, col=3)
fig2.add_trace(data32, row=1, col=4)
fig2.add_trace(data42, row=1, col=5)
fig2.add_trace(data52, row=1, col=6)
fig2.add_trace(data62, row=1, col=7)
fig2.update_traces(showscale=False, row=1, col=1)
fig2.update_traces(showscale=False, row=1, col=2)
fig2.update_traces(showscale=False, row=1, col=3)
fig2.update_traces(showscale=False, row=1, col=4)
fig2.update_traces(showscale=False, row=1, col=5)
fig2.update_traces(showscale=False, row=1, col=6)


fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0),

                            width=1500, height=150, showlegend=False,
                    xaxis=dict(showticklabels=False),
                    xaxis2=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    yaxis2=dict(showticklabels=False),
                    xaxis3=dict(showticklabels=False),
                    xaxis4=dict(showticklabels=False),
                    yaxis3=dict(showticklabels=False),
                    yaxis4=dict(showticklabels=False),
                    xaxis5=dict(showticklabels=False),
                    yaxis5=dict(showticklabels=False),
                    xaxis6=dict(showticklabels=False),
                    yaxis6=dict(showticklabels=False),
                    xaxis7=dict(showticklabels=False),
                    yaxis7=dict(showticklabels=False),)
fig2.show()
