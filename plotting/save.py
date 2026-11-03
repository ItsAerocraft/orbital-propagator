import os

import matplotlib.pyplot as plt


def save_pix(name):
    '''
    Saves the current figure to _docs/images/[name].png.

    Args:
        [name]: File name (without extension) to save the current figure as.
    '''
    os.makedirs("_docs/images", exist_ok = True)
    plt.savefig(f"_docs/images/{name}.png", dpi = 100)
