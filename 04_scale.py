# Scale مقیاس بندی

# There are different methods for scaling data, in this tutorial we will use a method called standardization.
# The standardization method uses this formula:
# z = (x - u) / s       z: new value  x: original value  u: mean  s: standard deviation

# StandardScaler()  it does the job for us.
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X = df[['Weight', 'Volume']]
scaledX = scale.fit_transform(X)

print(scaledX)


# Predict the CO2 emission from a 1.3 liter car that weighs 2300 kilograms:
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)

# scale.fit_transform(X)
# چون Scaler باید اول یاد بگیره مقیاس داده‌ها چطوره و بعد تبدیل کنه.

# برای داده جدید:
# scale.transform(new_data)
# چون Scaler قبلاً یاد گرفته؛ فقط باید داده جدید رو با همان مقیاس قبلی تبدیل کنه.

# پس:
# Training data:
# fit + transform

# New data:
# فقط transform

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

data = {
    'Area': [50, 80, 100, 150, 200, 250],
    'People': [1, 2, 3, 4, 5, 6],
    'Electricity': [120, 180, 220, 300, 390, 470]
}

df = pd.DataFrame(data)

X = df[['Area', 'People']]
y = df['Electricity']

scaledX = scale.fit_transform(X)

model = linear_model.LinearRegression()
model.fit(scaledX, y)

scaled = scale.transform([[120, 3]])

ElectricityPredict = model.predict([scaled[0]])
print(ElectricityPredict)

# چون مدل یاد گرفته که چطور بر اساس اون داده هایی که بهش دادیم، اونارو مقیاس بندی کنه
# پس بهش میگیم transform
# یعنی بر اساس چیزی که یاد گرفتی اینو مقیاس بندی کن
