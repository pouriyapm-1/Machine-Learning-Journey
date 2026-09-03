# Linear Regression:
# پیدا کردن یک خط از روی داده‌ها، برای اینکه بتوانیم مقدارهای جدید را پیش‌بینی کنیم


import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err  = stats.linregress(x,y)

def myfunc(x):
  return slope*x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x,y)
plt.plot(x, mymodel)
plt.show()

# It is important to know how the relationship between the values of the x-axis and the values of the y-axis is, if there are no relationship the linear regression can not be used to predict anything.
# This relationship - the coefficient of correlation - is called r.
# The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.
print(r)

# Predict Future Values
# Now we can use the information we have gathered to predict future values.

speed = myfunc(10) # predict the speed of a 10 years old car
print(speed)

# Example of very bad relationship
import matplotlib.pyplot as plt
from scipy import stats

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

print(r)
# it tells us that r = 0.013 it means the relationship is very bad and this data set is not suitable for linear regression.
