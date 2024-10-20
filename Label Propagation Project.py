from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
import numpy as np

# 生成數據
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_unlabeled, y_train, _ = train_test_split(X, y, test_size=0.9, random_state=42)
X_unlabeled, X_test, _, y_test = train_test_split(X_unlabeled, y, test_size=0.5, random_state=42)

# 構建標籤數據
y_unlabeled = -np.ones(X_unlabeled.shape[0])
X_combined = np.vstack((X_train, X_unlabeled))
y_combined = np.hstack((y_train, y_unlabeled))

# 標籤傳播
model = LabelPropagation()
model.fit(X_combined, y_combined)

# 評估模型
y_test_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))