"""Data visualization."""
# Data analysis packages:
# import pandas as pd
# import numpy as np

# Visualization packages:
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
# %matplotlib inline


def boxplot(dataset):
    """Make a box plot for each column of ``x``.

    Parameters
    ----------
    X : Array or a sequence of vectors

    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), sharey=False)

    axs[0].boxplot(dataset.abv, labels=['ABV'], notch=True, showmeans=True)
    axs[0].set_title('Alcohol By Volume')
    axs[1].boxplot(dataset.gravity, labels=['OG'], notch=True, showmeans=True)
    axs[1].set_title('Original Gravity')
    axs[2].boxplot(dataset.ibu, labels=['IBU'], notch=True, showmeans=True)
    axs[2].set_title('International Bitterness Units')
