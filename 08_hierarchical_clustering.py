# Heirarchical Clustering
# Cluster یعنی یک گروه از چیزهای شبیه به هم.

# Clustering یعنی:
# داده‌ها را بر اساس شباهتشان به گروه‌های مختلف تقسیم کنیم.

# Clustering → Unsupervised Learning

# Agglomerative = از پایین شروع می‌کنیم و گروه‌ها را کم‌کم با هم Merge می‌کنیم.

# Euclidean Distance
# یعنی فاصله‌ی معمولی بین دو نقطه.

# وقتی دو Cluster داریم، چطور تصمیم بگیریم این دو Cluster چقدر به هم نزدیک‌اند؟

# Ward Linkage:  سعی می‌کند بعد از Merge شدن، Clusterهای نسبتاً منسجم و کم‌پراکندگی داشته باشیم.

# Dendrogram در واقع نمودار تاریخچه‌ی Merge شدن Clusterهاست.

# ساختن و دیدن دندوگرام
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()

# --------------------
# تقسیم کردن داده ها به چند cluster نهایی و نشان دادن روی scatter plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

hierarchical_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

plt.scatter(x, y, c=labels)
plt.show()