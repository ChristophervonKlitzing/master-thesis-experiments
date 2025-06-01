from typing import List
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors 
from PIL import Image
import os

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from typing import List
import numpy as np

"""
plt.rcParams.update({
    "text.usetex": True,            # Use LaTeX to render all text
    "font.family": "sans-serif",         # Use serif fonts (LaTeX default)
    "font.serif": [],               # Empty means default Computer Modern
    "font.size": 11,                # Default LaTeX font size is 10-11pt
    "axes.labelsize": 25,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 25,
})
"""


def show_images_side_by_side(images: List[np.ndarray]):
    vmin = 0

    # Couldn't get exact vmap without changing the training script
    # The training-repository requires a refactoring, 
    # to not save the figures as pickled figures
    # but instead save the raw data. 8.5 matches visually and is 
    # therefore fine for illustration purposes.
    """
    vmax = 8.5
    n = len(images)
    fig = plt.figure(figsize=(n * 3, 3))
    # 1 row, n+1 columns; width_ratios for images and colorbar
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1]*n + [0.05], wspace=0.05)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))  # Adjust size as needed

    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        """
        if i == 0:
            ax.set_ylabel("$\psi_2$")
        ax.set_xlabel("$\phi_2$")
        """

    """
    # Create smaller colorbar axis (adjust position manually)
    cax = fig.add_subplot(gs[0, -1])
    # Shrink the colorbar height by setting its position manually
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height * (0.97 - 2 * 0.05)])  # shift up & shrink

    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    
    sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("free energy / $k_\mathrm{B} T$")
    """

    

    plt.tight_layout()
    plt.show()


def run(args):
    print("Create sequence of Ramachandran plots from an entropy-annealed run")

    dirpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "raw_ramachandrans")
    images: List[Image.Image] = []
    for f in sorted(os.listdir(dirpath)):
        if ".png" not in f:
            continue
        
        img_path = os.path.join(dirpath, f)
        img = Image.open(img_path)

        # Define the crop box: (left, upper, right, lower)
        crop_box = (43, 21, 982, 952)  # adjust to your needs
        cropped_img = img.crop(crop_box)
        images.append(cropped_img)

    show_images_side_by_side(images)
