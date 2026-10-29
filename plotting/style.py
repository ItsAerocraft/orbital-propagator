import matplotlib.pyplot as plt


def minorgridlines():
    '''
    Adds major and minor gridlines to the current matplotlib axes.
    '''
    plt.minorticks_on()
    plt.grid(True, which = "major", alpha = 0.3, linestyle = "--", linewidth = 1.5)
    plt.grid(True, which = "minor", alpha = 0.15, linestyle = ":", linewidth = 0.8)
