# Train/Test
# Train/Test is a method to measure the accuracy of your model.
# It is called Train/Test because you split the data set into two sets: a training set and a testing set.
# 80% for training, and 20% for testing.
# You train the model using the training set.
# You test the model using the testing set.

# Train the model means create the model.
# Test the model means test the accuracy of the model.

import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

plt.scatter(x, y)
plt.show()

# The training set should be a random selection of 80% of the original data.
# The testing set should be the remaining 20%.
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# Draw a polynomial regression line through the data points:
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

# How well does my training data fit in a polynomial regression?
import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))

print(r2)


# Let us find the R2 score when using testing data:
import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(test_y, mymodel(test_x))

print(r2)

# عملکرد روی داده‌هایی که مدل دیده
r2_score(train_y, mymodel(train_x))

# عملکرد روی داده‌هایی که مدل ندیده
r2_score(test_y, mymodel(test_x))

# Train → یادگیری
# Test → ارزیابی روی داده‌ی ندیده
# R² → اندازه‌گیری کیفیت عملکرد مدل

# a little exercise
import numpy as np
from sklearn.metrics import r2_score

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 5, 10, 17, 26, 37, 50, 65, 82, 101])

train_x = x[:8]
train_y = y[:8]

test_x = x[8:]
test_y = y[8:]

mymodel = np.poly1d(np.polyfit(train_x, train_y, 2))

R2_train = r2_score(train_y, mymodel(train_x))
print(R2_train) # 0.9999

R2_test = r2_score(test_y, mymodel(test_x))
print(R2_test) # 0.9964

#  چون میخواستیم ببینیم عملکرد مدل در مواقعی که داده رو ندیده (آموزش ندیده) چطوری هست
# برای همین y real رو با y ای که از test_x بدست اومده مقایسه میکنیم