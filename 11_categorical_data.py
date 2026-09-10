import pandas
from sklearn import linear_model

cars = pandas.read_csv("data.csv")
ohe_cars = pandas.get_dummies(cars[['Car']])  # One Hot Encoding: pd.get_dummies()

X = pandas.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1) # To combine the information, we can use concat()
y = cars['CO2']

regr = linear_model.LinearRegression()
regr.fit(X,y)

##predict the CO2 emission of a VW where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])

print(predictedCO2)


# Dummifying
import pandas as pd

colors = pd.DataFrame({'color': ['blue', 'red', 'green']})
dummies = pd.get_dummies(colors, drop_first=True)
dummies['color'] = colors['color']

print(dummies)

# برای تبدیل Categorical Data به داده‌ی عددی قابل استفاده در مدل‌های ML استفاده می‌شود.
# با pd.get_dummies() برای هر category یک ستون 0/1 ساخته می‌شود.
# با drop_first=True یک ستون حذف می‌شود؛ بنابراین برای n category → n−1 ستون داریم.
# category حذف‌شده نقش reference/baseline را دارد و وقتی تمام dummyها 0 باشند، یعنی همان category است.

