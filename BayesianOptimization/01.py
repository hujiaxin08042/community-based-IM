from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np


# 随机产生1000个分类数据集，10维特征，2个类别
x, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
# 不调参数的结果：
# rf = RandomForestClassifier()
# print(np.mean(cross_val_score(rf, x, y, cv=20, scoring='roc_auc')))

# 定义目标函数
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            max_depth=int(max_depth),
            random_state=2
        ),
        x, y, scoring='roc_auc', cv=5
    ).mean()
    return val

# 建立贝叶斯优化对象
# 第一个参数是要优化的目标函数，第二个参数是所需要的超参数名称，以及其范围
optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds={
        'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 15)
    },
    random_state=1234,
    verbose=2
)

optimizer.maximize(n_iter=10)
print("Final result:", optimizer.max)
