import matplotlib.pyplot as plt


def save_pix():
    '''
    Saves the current figures for each plot type to disk (if required).
    '''
    plt.savefig("orbit.png", dpi = 100)
    plt.savefig("altitude.png", dpi = 100)
    plt.savefig("silhouette.png", dpi = 100)
    plt.savefig("stochastic.png", dpi = 100)
    plt.close()
