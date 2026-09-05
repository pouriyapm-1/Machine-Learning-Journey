# Confusion Matrix
# It is a table that is used in classification problems to assess where errors in the model were made.
# The rows represent the actual classes the outcomes should have been. While the columns represent the predictions we have made. Using this table it is easy to see which predictions are wrong.

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()

# The matrix provides us with many useful metrics that help us to evaluate our classification model.
# The different measures include: Accuracy, Precision, Sensitivity (Recall), Specificity, and the F-score, explained below.+

# Accuracy: measures how often the model is correct.
Accuracy = metrics.accuracy_score(actual, predicted)

# Precision: Of the positives predicted, what percentage is truly positive?
Precision = metrics.precision_score(actual, predicted)

# Sensitivity (Recall):
# Of all the positive cases, what percentage are predicted positive?
# Sensitivity is good at understanding how well the model predicts something is positive:
Sensitivity_recall = metrics.recall_score(actual, predicted)

# Specificity:
# How well the model is at prediciting negative results?
# Specificity is similar to sensitivity, but looks at it from the perspective of negative results.
Specificity = metrics.recall_score(actual, predicted, pos_label=0)

# F-score:
# F-score is the "harmonic mean" of precision and sensitivity.
# It considers both false positive and false negative cases and is good for imbalanced datasets.
F1_score = metrics.f1_score(actual, predicted)

# show all metrics
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})


#                  Prediction
#                  0       1
#               ┌───────┬───────┐
# Actual     0  │  TN   │  FP   │
#               ├───────┼───────┤
#            1  │  FN   │  TP   │
#               └───────┴───────┘

# سطرها = واقعیت (Actual)
# ستون‌ها = پیش‌بینی مدل (Prediction)

# مثلاً اگر داخل FP عدد 30 باشد:

# 30 نفر واقعاً 0 بودند، ولی مدل اشتباهاً آنها را 1 تشخیص داده.

# Accuracy   → چندتا کلاً درست؟
# Precision  → از مثبت‌هایی که گفتم، چندتا واقعاً مثبت؟
# Recall     → از مثبت‌های واقعی، چندتا رو پیدا کردم؟
# Specificity→ از منفی‌های واقعی، چندتا رو درست پیدا کردم؟
# F1         → تعادل Precision و Recall