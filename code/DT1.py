import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization

import os

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("/data/DT1.csv")

X = df.drop(['accuracy_cat'], axis=1)

y = df['accuracy_cat']

X_train = pd.get_dummies(X)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy", max_depth=10)
dt = dt.fit(X_train,y)

X_train = X_train.rename(columns={"imp_miss_val_Summary statistics (e.g mean, median, mode)": "imp_miss_val_summ_stat", "outliers_Remove": "outliers_rm", "imp_miss_val_Drop rows": "imp_miss_val_drop_rows"})

feature_cols=list(X_train.columns.values)

import six
import sys
sys.modules['sklearn.externals.six'] = six

import sklearn
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['above','below'], precision=1)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

graph.write_png("/results/DT_1.png")