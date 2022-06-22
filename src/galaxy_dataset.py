import torch
import numpy as np
import matplotlib.pyplot as plt

class GalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target.type(torch.LongTensor) 
        self.transform = transform
        
        self.c = len(np.unique(target))
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)
    
def norm(vals, vmin=None, vmax=None, Q=8, stretch=None):
    """
    For visualization purposes normalize image with `arcsinh((vals-vmin)/(vmax-vmin)), 
    with vals either specified or within 0.01 and 0.99 quantiles of all values. 
    
    Q and stretch control the arcsinh softening parameter, see Lupton et al. 2004 and
    see https://docs.astropy.org/en/stable/_modules/astropy/visualization/lupton_rgb.html#make_lupton_rgb
    """
    if vmin is None: vmin = np.quantile(vals, 0.01)
    if vmax is None: vmax = np.quantile(vals, 0.99)
    
    if stretch is None:
        return np.arcsinh(Q*(vals - vmin) / (vmax-vmin)) / Q
    else:
        return np.arcsinh(Q*(vals - vmin) / stretch) / Q
    
def show_images(dataset, N_images=16):
    """Show first 16 images of a dataset (e.g. the training images and labels).
    """
    fig = plt.figure(figsize=(8, 8), dpi=150) 

    for i, (image, cls) in enumerate(zip(*dataset[:16])):
        ax = fig.add_subplot(4, 4, i+1)

        image = norm(image, Q=5).permute(1, 2, 0).clip(0, 1)

        ax.imshow(image, aspect='equal')
        ax.text(0.5, 0.85, "Merger" if cls == 1 else "No merger", ha='center', color='white', transform=ax.transAxes)

        ax.axis('off')

    plt.show()