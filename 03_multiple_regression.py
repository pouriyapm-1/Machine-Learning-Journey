# Multiple Regression
# Multiple regression is like linear regression, but with more than one independent value,
# meaning that we try to predict a value based on two or more variables.

import pandas as pd
from sklearn import linear_model

df = pd.read_csv('carsdata.csv')

X = df[['Weight','Volume']]
y = df['CO2']
#* It is common to name the list of independent values with a upper case X, and the list of dependent values with a lower case y.

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[2300,1300]])
print(predictedCO2)

# Coefficient (ضریب)
print(regr.coef_)
# These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
# And if the engine size (Volume) increases by 1cm3, the CO2 emission increases by 0.00780526g.
#* (با ثابت نگه داشتن متغیر های دیگر، اگر یک متغیر رو یک واحد بیشتر کنیم با این ضریب  مقدار خروجی افزایش پیدا میکنه

# قرارداد:
# X = features
# y = target