import matplotlib.pyplot as plt
from collections import Counter
import ast
import numpy as np
import matplotlib.style as style

style.use("seaborn-v0_8-whitegrid") #sets the size of the charts
style.use('ggplot')

with open("files/wiki_labels_r_p.txt", "r") as wiki, open("files/tabtest_labels_r_p.txt", "r") as test:
    wiki = wiki.readlines()
    wiki = [ast.literal_eval(i) for i in wiki]
    wiki = list(set(wiki))
    wiki = [str((i[1], i[2])) for i in wiki]
    counter_wiki = dict(Counter(wiki))
    counter_wiki = dict(sorted(counter_wiki.items(), key=lambda x:x[1], reverse=True))
    counter_wiki = {k:v for k,v in counter_wiki.items() if v >5}


    test = test.readlines()
    test = [ast.literal_eval(i) for i in test]
    test = list(set(test))
    test = [str((i[1], i[2])) for i in test]
    counter_test = dict(Counter(test))
    counter_test = dict(sorted(counter_test.items(), key=lambda x:x[1], reverse=True))
    counter_test = {k:v for k,v in counter_test.items() if v >5}

    pairs = ["('MISC', 'DEM')", "('ORG', 'MISC')", "('MISC', 'ORG')", "('DEM', 'MISC')", "('ORG', 'DEM')", "('MISC', 'QUANTITY')", "('LOC', 'ORG')", "('ORG', 'LOC')", "('PERSON', 'DEM')", "('ORG', 'PERSON')", "('ORG', 'QUANTITY')", "('MISC', 'PERSON')", "('QUANTITY', 'DATETIME')", "('LOC', 'MISC')"]
    #Add these for individual pairs too-> "('PERSON', 'MISC')", "('DEM', 'DATETIME')", "('MISC', 'DATETIME')", "('DEM', 'ORG')", "('CODE', 'QUANTITY')", "('PERSON', 'ORG')", "('QUANTITY', 'MISC')", "('DEM', 'QUANTITY')", "('MISC', 'CODE')", "('LOC', 'DEM')", "('QUANTITY', 'DEM')"]

    wiki_values = [counter_wiki[i] if i in counter_wiki else 0 for i in pairs]
    tab_test_values = [counter_test[i] if i in counter_test else 0 for i in pairs]

    N = 2
    width = 0.3

    plt.figure(figsize = (10, 10))

    ind = np.arange(len(pairs))
    plt.barh(ind-width, wiki_values, width, label = 'Wiki')
    plt.barh(ind, tab_test_values, width, label = 'Tab test')
    plt.gca().invert_yaxis()
    plt.yticks(ind, pairs, fontsize = 8)
    plt.legend(loc = 'lower right', frameon = True)
    plt.tight_layout()
    plt.savefig('label_confusion.png', dpi = 600)
    plt.show()




