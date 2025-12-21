# read from csv pandas housing.csv
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

housing = pd.read_csv("datasets/housing/housing.csv")

imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
print(imputer.statistics_)
X = imputer.transform(housing_num)
