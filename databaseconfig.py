import pandas as pd

train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
new_cols = ['Id', 'SalePrice', 'OverallQual', 'FullBath', 'YearBuilt', 'GrLivArea', 'GarageCars']
train = train[new_cols]
