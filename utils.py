
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import numpy as np 
import glob
from PIL import Image

def sample_gaussian(m, v):
	
	sample = torch.randn(m.shape).to(device)
	z = m + (v**0.5)*sample
	return z



def make_gif(frame_folder,path_save):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"),key=len)]
    frame_one = frames[0]
    frame_one.save(path_save, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)


def plot(x,labels,class_dict=None,legend = "full", epoch = None,save = None):
    xl = "First dimension"
    if epoch is not None:
        xl = xl + " epoch " + str(epoch)
    df = pd.DataFrame(x,columns=[xl,"Second dimension"])

    df['y'] = [str(labels[i]) for i in range(len(labels))] if class_dict==None else [class_dict[str(labels[i])] for i in range(len(labels))]
    plt.figure(figsize=(16,10))

    
    sns.scatterplot(
    x=xl, y="Second dimension",
    hue="y",
    palette=sns.color_palette("bright", len(np.unique(labels))),
    data=df,
    legend=legend,
    alpha=0.3
    )
    
    if save is not None:
        plt.savefig(save,dpi=200)
        plt.close()


def plotfm(model,data,liste,legend = "full", epoch = None, eps = None,save = None):
    if not model.encoder.sample:
        m,v = model.encoder(data)
        X = model.encoder.sampling(m,v,eps)
    else:
        _,m,v = model.encoder(data)
        X = model.encoder.sampling(m,v,eps)
    X = X.cpu().detach()

    plot(x = X,labels = liste,legend = legend,epoch=epoch,save=save)
    
def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return(params)


