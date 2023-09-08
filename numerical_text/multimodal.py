import pandas as pd
import autogluon
from autogluon.multimodal import MultiModalPredictor
import warnings
import numpy as np
import ast
import tqdm
import pandas as pd
import autogluon
from tabulate import tabulate
import random
import json
import ast
import numpy as np
import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)


def average(lst):
    return sum(lst) / len(lst)

if __name__=="__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

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

    df1['semantic_type'] = df1['semantic_type'].map(one_hot)

    df1_ = df1[['semantic_type', 'span_text', 'no_tokens', 'no_subtokens', 'min', 'max', 'average', 'median', 'sum', 'label']]
    df2_ = df2[['semantic_type', 'span_text','no_tokens', 'no_subtokens',  'min', 'max', 'average', 'median', 'sum', 'label']]


    train = df1_
    test = df2_

    label = 'label'

    predictor = MultiModalPredictor(label=label,  verbosity = 4)

    predictor.fit(train,hyperparameters={"data.categorical.convert_to_text": False})

    y_test = test[label]
    test_data_nolab = test.drop(columns=[label])
    test_data_nolab.head()
    y_pred = predictor.predict(test_data_nolab)

    df2['predictions']= y_pred
    df2.to_csv('train_test_wiki.csv', index = False)