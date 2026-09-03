# Polynomial Regression
# If your data points clearly will not fit a linear regression (a straight line through all data points), it might be ideal for polynomial regression.
# Polynomial regression, like linear regression, uses the relationship between the variables x and y to find the best way to draw a line through the data points.

import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(np.polyfit(x, y, 3))

myline = np.linspace(1, 22, 100)

plt.scatter(x,y)
plt.plot(myline, mymodel(myline))
plt.show()

# R-squared
# میگه مدل چقد خوب فیت شده
# it's between 0 and 1
import numpy as np
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(np.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))

# r --> بررسی رابطه x,y
# R2 --> real y vs predicted y


# Exercises (Linear & Polynomial Regression)
#1
import matplotlib.pyplot as plt
from scipy import stats

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x,y)
plt.plot(x, mymodel)
plt.show()

print(r)
print(myfunc(6))

#2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = [1, 2, 3, 4, 5, 6, 7]
y = [3, 5, 10, 17, 26, 37, 50]

mymodel = np.poly1d(np.polyfit(x, y, 2))

myline = np.linspace(1, 7, 100)

plt.scatter(x,y)
plt.plot(myline, mymodel(myline))
plt.show()

print(r2_score(y, mymodel(x))) # آره مدل واقعا مناسب هست. چون R2 عدد 0.99 رو میده
# به این معنا که y predicted ارتباط خیلی قوی با y real داره

print(mymodel(8))  # 65.42 پیشبینی میکنه مقدار y رو