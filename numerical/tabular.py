import pandas as pd
import autogluon
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from tabulate import tabulate
import ast
import numpy as np
import json


def average(lst):
    return sum(lst) / len(lst)

one_hot = {'PERSON':1, 'ORG':2, 'LOC':3, 'DATETIME':4, 'QUANTITY':5, 'DEM':6, 'MISC':7, 'CODE':8}

df1 = pd.read_csv('../data/wiki_train.csv')
df2 = pd.read_csv('../data/wiki_test.csv')

df1['probability'] = df1.probability.apply(lambda x: ast.literal_eval(x))
df2['probability'] = df2.probability.apply(lambda x: ast.literal_eval(x))

df1['median'] = df1.probability.apply(lambda x: np.median(np.array(x)))
df2['median'] = df2.probability.apply(lambda x: np.median(np.array(x)))

df1['sum'] = df1.probability.apply(lambda x: np.sum(np.array(x)))
df2['sum'] = df2.probability.apply(lambda x: np.sum(np.array(x)))

df1['min'] = df1.probability.apply(lambda x: min(x))
df2['min'] = df2.probability.apply(lambda x: min(x))

df1['max'] = df1.probability.apply(lambda x: max(x))
df2['max'] = df2.probability.apply(lambda x: max(x))

df1['average'] = df1.probability.apply(lambda x: average(x))
df2['average'] = df2.probability.apply(lambda x: average(x))


labels = [one_hot[el] for el in df1['semantic_type'].tolist()]
labels2 = [one_hot[el] for el in df2['semantic_type'].tolist()]

df1['numeric_types'] = labels
df2['numeric_types'] = labels2

df1_ = df1[['numeric_types', 'min', 'max', 'average', 'median', 'sum', 'label']]
df2_ = df2[['numeric_types', 'min', 'max', 'average', 'median', 'sum', 'label']]


train = df1_
dev = df2_

label = 'label'

predictor = autogluon.tabular.TabularPredictor(label=label,  eval_metric = 'f1', verbosity = 1, learner_kwargs={'positive_class':'MASK'})
predictor.fit(train)

y_dev = dev[label]
dev_data_nolab = dev.drop(columns=[label])
dev_data_nolab.head()
y_dev = predictor.predict(dev_data_nolab)

df2['predictions']= y_dev
df2.to_csv('tab_train_wiki_test.csv', index = False)

