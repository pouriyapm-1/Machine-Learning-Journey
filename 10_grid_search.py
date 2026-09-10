# Grid Search
# The majority of machine learning models contain parameters that can be adjusted to vary how the model learns.
# For example, the logistic regression model, from sklearn, has a parameter C that controls regularization,which affects the complexity of the model.

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris['data']
y = iris['target']

logr = LogisticRegression(max_iter=10000)

C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
scores = []

for choice in C:
  logr.set_params(C=choice)
  logr.fit(X, y)
  scores.append(logr.score(X, y))

print(scores)

# Grid Search 
# یعنی امتحان کردن ترکیب‌های مختلف hyperparameterها و انتخاب مقداری که بهترین عملکرد را روی داده‌ی ارزیابی داشته باشد.
# و C، max_iter و این چیزها پارامترهایی هستند که ما قبل از fit() تعیین می‌کنیم؛ به این‌ها معمولاً hyperparameter می‌گیم.