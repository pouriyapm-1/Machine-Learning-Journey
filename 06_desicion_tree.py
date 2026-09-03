# Decision Tree
# To make a decision tree, all data has to be numerical.
# Pandas has a map() method that takes a dictionary with information on how to convert the values.

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data = {
  'Age': [36,42,23,52,43,44,66,35,52,35,24,18,45],
  'Experience': [10,12,4,4,21,14,3,14,13,5,3,3,9],
  'Rank': [9,4,6,4,8,5,7,9,7,9,5,7,9],
  'Nationality': ["UK","USA","N","USA","USA","UK","N","UK","N","N","USA","UK","UK"],
  'Go': ["NO","NO","NO","NO","YES","NO","YES","YES","YES","YES","NO","YES","YES"]
}

df = pd.DataFrame(data)

d = {'UK':0, 'USA':1, 'N':2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES':1, 'NO':0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)
print(dtree.predict([[40,10,7,1]]))
print(dtree.predict([[40, 10, 6, 1]]))

# Different Results
# You will see that the Decision Tree gives you different results if you run it enough times, even if you feed it with the same data.
# That is because the Decision Tree does not give us a 100% certain answer. It is based on the probability of an outcome, and the answer will vary.

# درخت دائماً این سؤال را می‌پرسد:
# «بهترین سؤال برای جدا کردن این داده‌ها چیه؟»

# مفاهیم درخت تصمیم
# 1. Node
# هر جایی که یک سؤال پرسیده می‌شود:
# Rank <= 6.5?

# 2. Split
# تقسیم کردن داده‌ها بر اساس آن سؤال:
# Rank <= 6.5     |     Rank > 6.5

# 3. Gini
# معیار سنجش اینکه یک گروه چقدر مخلوط است.
# Gini = 0
# یعنی کاملاً خالص.

# 4. Samples
# تعداد داده‌هایی که در آن نقطه باقی مانده‌اند.
# samples = 8
# یعنی 8 نمونه در آن قسمت درخت داریم.

# 5. Value
# تعداد هر کلاس:
# value = [1, 7]
# یعنی:
# NO = 1
# GO = 7

# مدل داده‌ها را با سؤال‌های مختلف split می‌کند و با معیارهایی مثل Gini دنبال splitهایی می‌گردد که کلاس‌ها را بهتر از هم جدا کنند.