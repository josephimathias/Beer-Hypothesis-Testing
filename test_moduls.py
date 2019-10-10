"""This module contains functions used to perform hypotesis testing."""

# Data analysis packages:
# import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns


def welch_t(a, b):
    """Return Welch's t statistic for two samples."""
    # welch t calculation
    numerator = a.mean() - b.mean()
    # “ddof = Delta Degrees of Freedom”: the divisor used in N- ddof,
    #  where N represents the number of elements. By default ddof is zero.
    denominator = np.sqrt(a.var(ddof=1)/a.size + b.var(ddof=1)/b.size)

    return np.abs(numerator/denominator)


def welch_df(a, b):
    """Return the effective degrees of freedom for two samples welch test."""
    # calculate welch df
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = a.size, b.size

    numerator = (s1/n1 + s2/n2)**2
    denominator = (s1 / n1)**2/(n1 - 1) + (s2 / n2)**2/(n2 - 1)
    df = numerator/denominator
    return df


def p_value_welch_ttest(a, b, two_sided=False, alpha=0.05):
    """Return the p-value for Welch's t-test given two samples."""
    """
    By default, the returned p-value is for a one-sided t-test.
    Set the two-sided parameter to True if you wish to perform a two-sided
    t-test instead.
    """

    t_welch = welch_t(a, b)
    df = welch_df(a, b)

    # Calculate the critical t-value
    t_crit = stats.t.ppf(1-alpha/2, df=df)
    p_welch = (1-stats.t.cdf(np.abs(t_welch), df)) * 2

    # return results
    if (t_welch > t_crit and p_welch < alpha):
        print(f"""Null hypohesis rejected. Results are statistically significant with:
            t_value = {round(t_welch, 4)}, 
            critical t_value = {round(t_crit, 4)}, and 
            p-value = {round(p_welch, 4)}""")
    else:
        print(f"""We fail to reject the Null hypothesis with: 
        t_value = {round(t_welch, 4)}, 
        critical t_value = {round(t_crit, 4)}, 
        and p_value = {round(p_welch, 4)}""")

    if two_sided:
        return 2 * p_welch
    else:
        return p_welch

# Two sample t-test


def sample_variance(sample):
    """Calculates sample varaiance."""
    sample_mean = np.mean(sample)
    return np.sum((sample-sample_mean)**2)/(len(sample) - 1)


def pooled_variance(sample1, sample2):
    """retun pooled sample variance."""
    n_1, n_2 = len(sample1), len(sample2)
    var_1, var_2 = sample_variance(sample1), sample_variance(sample2)

    return ((n_1 - 1) * var_1 + (n_2 - 1) * var_2)/(n_1 + n_2 - 2)


def twosample_tstatistic(expr, ctrl, alpha=0.05):
    """Plot two_tailed t-distibution."""
    # Visualize sample distribution for normality
    sns.set(color_codes=True)
    sns.set(rc={'figure.figsize': (12, 10)})

    exp_mean, ctrl_mean = np.mean(expr), np.mean(ctrl)
    pool_var = pooled_variance(expr, ctrl)
    n_e, n_c = len(expr), len(ctrl)
    num = exp_mean - ctrl_mean
    denom = np.sqrt(pool_var * ((1/n_e) + (1/n_c)))

    # Calculate the critical t-value
    df = len(expr) + len(ctrl) - 2
    t_crit = stats.t.ppf(1-alpha/2, df=df)

    t_stat = num/denom
    p_value = (1 - stats.t.cdf(t_stat, df=df)) * 2
    # p_value =  stats.t.cdf(t_stat, df=df)

    # return results
    if (t_stat > t_crit and p_value < alpha):

        print("Null hypohesis rejected. Results are statistically significant with:", "\n", 
              "t-statistic =", round(t_stat, 4), "\n",
              "critical t-value =", round(t_crit, 4), "and", "\n"
              "p-value =", round(p_value, 4), '\n')
    else:
        print("Null hypothesis is True with:", "\n",
              "t-statistic =", round(t_stat, 4), "\n", 
              "critical t-value =", round(t_crit, 4), "and", "\n",
              "p-value = ", round(p_value, 4), '\n')
        
        
        print(">> We fail to reject the Null Hypothesis", "\n")
        
    print('----- Groups info  -----')
    print(f"""The groups contain {len(expr)} , and {len(ctrl)} observations. 
    The means are {np.round(exp_mean, 4)}, and {np.round(ctrl_mean, 4)} respectivelly""")

    return t_stat
