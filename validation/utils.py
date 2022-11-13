import pandas as pd
import requests



def read_allowed_numbers(sheet_id='1eVjmjribNgYRRcG9i6lcztRLdC8LiB9jgLc85BiOSbU'):
    r = requests.get(f'https://docs.google.com/spreadsheet/ccc?key={sheet_id}&output=csv')
    open('dataset.csv', 'wb').write(r.content)
    df = pd.read_csv('dataset.csv')
    return df['Allowed'].str.lower().unique()

