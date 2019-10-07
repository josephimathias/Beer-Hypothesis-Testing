# flatiron_stats
import numpy as np
import scipy.stats as stats
import seaborn as sns


def welch_t(a, b):
    """ Calculate Welch's t statistic for two samples. """

    numerator = a.mean() - b.mean()

    # “ddof = Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
    #  where N represents the number of elements. By default ddof is zero.

    denominator = np.sqrt(a.var(ddof=1)/a.size + b.var(ddof=1)/b.size)

    return np.abs(numerator/denominator)


def welch_df(a, b):
    """ Calculate the effective degrees of freedom for two samples. This function returns the degrees of freedom """

    s1 = a.var(ddof=1)
    s2 = b.var(ddof=1)
    n1 = a.size
    n2 = b.size

    numerator = (s1/n1 + s2/n2)**2
    denominator = (s1 / n1)**2/(n1 - 1) + (s2 / n2)**2/(n2 - 1)

    return numerator/denominator


def p_value_welch_ttest(a, b, two_sided=False):
    """Calculates the p-value for Welch's t-test given two samples.
    By default, the returned p-value is for a one-sided t-test.
    Set the two-sided parameter to True if you wish to perform a two-sided t-test instead.
    """
    alpha = 0.05
    t = welch_t(a, b)
    df = welch_df(a, b)
    # Calculate the critical t-value
    t_crit = stats.t.ppf(1-alpha, df=df)

    p = 1-stats.t.cdf(np.abs(t), df)

    # return results
    if (t > t_crit and p < alpha):
        print("Null hypohesis rejected. Results are statistically significant with t-value = ", t,
              "critical t-value = ", t_crit, "and p-value = ", p)
    else:
        print('Null hypothesis is True with t-value = ', t,
              "critical t-value = ", t_crit, 'and p-value = ', p)

    if two_sided:
        return 2*p
    else:
        return p


def ttest(sample, popmean, alpha):

    # Visualize sample distribution for normality
    sns.set(color_codes=True)
    sns.set(rc={'figure.figsize': (12, 10)})
    sns.distplot(sample)

    # Population mean
    mu = popmean

    # Sample mean (x̄) using NumPy mean()
    x_bar = round(sample.mean(), 2)

    # Sample Standard Deviation (sigma) using Numpy
    sigma = round(np.std(sample, ddof=1), 3)

    # Degrees of freedom
    df = len(sample) - 1

    # Calculate the critical t-value
    t_crit = stats.t.ppf(1-alpha, df=df)
    # (x_bar - mu)/(sigma/np.sqrt(len(sample)))

    # Calculate the t-value and p-value
    results = stats.ttest_1samp(a=sample, popmean=mu)
    t_value = round(results[0], 2)
    p_value = np.round((results[1]), 4)

    # return results
    if (t_value > t_crit and p_value < alpha):
        print("Null hypohesis rejected. Results are statistically significant with t-value = ", t_value,
              "critical t-value = ", t_crit, "and p-value = ", p_value)
    else:
        print('Null hypothesis is True with t-value = ', t_value,
              "critical t-value = ", t_crit, 'and p-value = ', p_value)

    print('The sample contains', len(sample), 'observations, having a mean of', x_bar, 'and a standard devation (sigma) =', sigma,
          ', with', df, 'degrees of freedom. The difference between sample and population means is:', (x_bar - mu))
    return t_crit
