"""Data cleaning file."""
# import numpy as np


def chose_only_hypothesis_colums(df):
    """Select olny columns that are used in this hypohesis testing."""
    lst = ['abv', 'ibu', 'gravity', 'abv_min', 'abv_max', 'ibu_min',
           'ibu_max', 'srm_min', 'srm_max', 'og_min', 'fg_min', 'fg_max']
    return df[lst]


def beer_vital_stats(beers):
    """Return the three beer vital stats.

    input: dataframe
    output: dataframe containing the three columns
    """
    lst = ['abv', 'ibu', 'gravity']
    beer_vital_stats = beers[lst]
    print('Dataset before removing null records:', beer_vital_stats.shape)
    idx = (beer_vital_stats.notnull()).all(axis=1)
    beer_vital = beer_vital_stats[idx]
    print('Dataset after removing all the null records:', beer_vital.shape)

    return beer_vital
