import torch
from utilities3 import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from plotly.subplots import make_subplots
import plotly.graph_objects as go

PATH = '/home/liux0t/FMM/model/MgNet_DCdarcy20c62023-09-06 20:23:08.383496.pt' #'/home/liux0t/FMM/model/MgNet2darcy20c62023-08-28.pt' #
model = torch.load(PATH)
model = torch.load(PATH)
model.eval()
dataOpt = {}
dataOpt['data'] = 'darcy20c6'
dataOpt['GN'] = False
dataOpt['sampling_rate'] = 2
dataOpt['dataSize'] = {'train': range(1280), 'test': range(112), 'val':range(10), 'jin':range(3)}
dataOpt['sample_x'] = True
dataOpt['normalizer_type'] = 'GN'
dataOpt['batch_size'] = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_train, y_train, x_normalizer, y_normalizer = getDarcyDataSet(dataOpt, flag='train', return_normalizer=True)
x_test, y_test = getDarcyDataSet(dataOpt, flag='test', return_normalizer=False, normalizer=x_normalizer)
x_test_mg = x_test.unsqueeze(1).to(device)
x_test_mwt = x_test.squeeze()[:, ..., np.newaxis]



PATH = '/home/liux0t/FMM/model/MWTdarcy20c62023-09-19 11:39:50.406213.pt'
model_mwt = torch.load(PATH)
model_mwt.eval()
model_mwt = model_mwt.to(device)
PATH = '/home/liux0t/FMM/model/Unetdarcy20c62023-09-19 12:11:26.375135.pt'
model_unet = torch.load(PATH)
model_unet.eval()
model_unet = model_unet.to(device)
PATH = '/home/liux0t/FMM/model/FNOdarcy20c62023-09-25 20:23:18.268336.pt'
model_fno = torch.load(PATH)
model_fno.eval()
model_fno = model_fno.to(device)
PATH = '/home/liux0t/FMM/model/UNOdarcy20c62023-01-24 06:24:18.852040.pt'
model_uno = torch.load(PATH)
model_uno.eval()
model_uno = model_uno.to(device)
y_pred_mg = model(x_test_mg[0:1, ...]).squeeze()
y_pred_mwt = model_mwt(x_test_mwt[0:1, ...].to(device)).squeeze()
y_pred_unet = model_unet(x_test_mg[0:1, ...].to(device)).squeeze()
y_pred_uno = model_uno(x_test_mwt[0:1, ...].to(device)).squeeze()
y_pred_fno = model_fno(x_test_mwt[0:1, ...].to(device)).squeeze()

data_a = go.Contour(z=x_test_mg[0, ...].squeeze().detach().cpu(),
colorscale='RdBu_r',
zmax=15,
zmin=1,
# line_smoothing=0.85,
# line_width=0.1,
contours=dict(
    coloring='heatmap',),
)


data0=go.Contour(z=y_test[0, ...].detach().cpu(),
colorscale='RdBu_r',
zmax=0.015,
zmin=0,
# line_smoothing=0.85,
# line_width=0.1,
contours=dict(
    coloring='heatmap',),
)

data1=go.Contour(z=y_pred_mg.detach().cpu(),
colorscale='RdBu_r',
zmax=0.015,
zmin=0,
# line_smoothing=0.85,
# line_width=0.1,
contours=dict(
   coloring='heatmap',),
)

data2=go.Contour(z=y_pred_mwt.detach().cpu(),
colorscale='RdBu_r',
zmax=0.015,
zmin=0,
# line_smoothing=0.85,
# line_width=0.1,
contours=dict(
   coloring='heatmap',),
)

data3=go.Contour(z=y_pred_unet.detach().cpu(),
colorscale='RdBu_r',
zmax=0.015,
zmin=0,
# line_smoothing=0.85,
# line_width=0.1,
contours=dict(
    coloring='heatmap',),
)
data4=go.Contour(z=y_pred_fno.detach().cpu(),
colorscale='RdBu_r',
zmax=0.015,
zmin=0,
# line_smoothing=0.85,
# line_width=0.1,
contours=dict(
    coloring='heatmap',),
)


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

fig = make_subplots(rows=1, cols=5, subplot_titles=("a","Groud truth u","MgNO", "MWT", "UNet",), horizontal_spacing=0.02) #vertical_spacing = 0.05
# add title for each subplot
fig.add_trace(data_a, row=1, col=1,)
fig.add_trace(data0, row=1, col=2,)
fig.add_trace(data1, row=1, col=3,)
# fig.add_trace(colorbar_trace)
fig.add_trace(data2, row=1, col=4)
fig.add_trace(data3, row=1, col=5)
# fig.add_trace(data4, row=1, col=5)
# fig = make_subplots(rows=1, cols=1)
# fig.add_trace(data1, row=1, col=1,)
fig.update_traces(showscale=False, row=1, col=1)
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
                 )

fig.show()
fig.write_image('model/figures4paper/predict.png', scale=2)

fig = make_subplots(rows=1, cols=5, subplot_titles=("MgNO", "MWT", "UNet", "FNO","UNO"), horizontal_spacing=0.02) #vertical_spacing = 0.05
data0=go.Contour(z=y_test[0, ...]-y_pred_mg.detach().cpu(),
colorscale='RdBu_r',
zmax=0.0006,
zmin=-0.0006,
contours=dict(  
    coloring='heatmap',),
)
data1=go.Contour(z=y_test[0, ...]-y_pred_mwt.detach().cpu(),
colorscale='RdBu_r',
zmax=0.0006,
zmin=-0.0006,
contours=dict(
    coloring='heatmap',),
)
data2=go.Contour(z=y_test[0, ...]-y_pred_unet.detach().cpu(),
colorscale='RdBu_r',
zmax=0.0006,
zmin=-0.0006,
contours=dict(
    coloring='heatmap',),
)
data3=go.Contour(z=y_test[0, ...]-y_pred_fno.detach().cpu(),
colorscale='RdBu_r',
zmax=0.0006,
zmin=-0.0006,
contours=dict(
    coloring='heatmap',),
)
data4=go.Contour(z=y_test[0, ...]-y_pred_uno.detach().cpu(),
colorscale='RdBu_r',
zmax=0.0006,
zmin=-0.0006,
contours=dict(
    coloring='heatmap',),
)

fig.add_trace(data0, row=1, col=1,) 
fig.add_trace(data1, row=1, col=2,)
fig.add_trace(data2, row=1, col=3)
fig.add_trace(data3, row=1, col=4)
fig.add_trace(data4, row=1, col=5)
fig.update_traces(showscale=False, row=1, col=1)
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
)
fig.show()
fig.write_image('model/figures4paper/error.png', scale=1)


