"""Data visualization."""

# Visualization packages:
import numpy as np
import seaborn as sns
import scipy.stats as stats
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


def visualize_t(t_stat, n1, n2, alpha=0.05):
    """Plot t-distrubtion."""

    # initialize a matplotlib "figure"
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()

    # generate points on the x axis between -4 and 4:
    xs = np.linspace(-4, 4, 500)
    # use stats.t.pdf to get values on the pdf for the t-distribution
    ys = stats.t.pdf(xs, (n1 + n2 - 2), 0, 1)
    ax.plot(xs, ys, linewidth=3, color='darkred')

    t_crit = round(stats.t.ppf(1-alpha/2, df=(n1+n2)-2), 4)

    ax.axvline(t_stat, color='red', linestyle='--', lw=2,
               label='t-statistic ')
    ax.axvline(t_crit, color='green', linestyle='--', lw=2,
               label='t-critical')
    # ax.axvline(-t_stat, color='black', linestyle='--', lw=4)
    ax.axvline(-t_crit, color='green', linestyle='--', lw=2)
    ax.fill_betweenx(ys, xs, t_crit, where=xs > t_crit,
                     color='#376cb0', alpha=0.7)
    ax.fill_betweenx(ys, xs, -t_crit, where=xs < -t_crit,
                     color='#376cb0', alpha=0.7)
    plt.title('Two-sided t-distibution', size=20)
    plt.xlabel('t (p, df)', size=15)
    plt.ylabel('Probability Distribution', size=15)

    ax.legend()

    # confidence interval
#     conf = stats.t.interval(alpha=1-alpha, df=(n1+n2) -2, loc, scale = )

    # Draw two sided boundary for critical-t
    plt.show()
    return None


# conf = stats.t.interval(alpha = 0.95,                       # Confidence level
#                  df= len(sample)-1,                  # Degrees of freedom
#                  loc = x_bar,                        # Sample mean
#                  scale = sd)
# print('sample mean is:', x_bar, 'and the confidence interval is', conf)