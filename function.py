def coast_list(df):
    regions = []
    wc = ['CA', 'CO', 'WA', 'OR', 'AK']
    ec = ['FL', 'NY', 'MD', 'GA', 'DC', 'VA', 'SC', 'NC', 'CT', 'DE', 'NJ', 'VT', 'ME', 'NH', 'MA', 'RI', 'PA']
    for state in df.state:
        if state in wc:
            regions.append('wc')
        elif state in ec:
            regions.append('ec')
        else:
            regions.append(None)
    df['region'] = regions
    return df[['beer_name', 'brewery_name', 'ibu', 'state', 'region']].dropna()