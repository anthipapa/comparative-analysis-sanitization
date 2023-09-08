import json
import pandas as pd
import ast
import glob, os

## All
# os.chdir("domain/")
# for file in sorted(glob.glob("*.csv"), key=os.path.getmtime):

#     df1 = pd.read_csv(file)
#     df1 = df1.iloc[:, 1:]
#     name = str(file[:-4])

#     df1 = df1.loc[df1['predictions'] == 'MASK']

#     keys = list(set(df1['doc_id']))

#     out = {}
#     for key in keys:
#         df = df1.loc[df1['doc_id'] == key]
#         spans = list(df['offsets'])
#         spans = [ast.literal_eval(i) for i in spans]
#         out[key] = spans


#     f_out = open(name+'.json', "w")
#     json.dump(out, f_out)
#     f_out.close()

    
## Single file
file = "domain_test/tab_train_wiki_test.csv"

df1 = pd.read_csv(file)
df1 = df1.iloc[:, 1:]

df1 = df1.loc[df1['predictions'] == 'MASK']

keys = list(set(df1['doc_id']))

out = {}
for key in keys:
    df = df1.loc[df1['doc_id'] == key]
    spans = list(df['offsets'])
    spans = [ast.literal_eval(i) for i in spans]
    out[key] = spans


f_out = open('domain_test/tab_train_wiki_test.json', "w")
json.dump(out, f_out)
f_out.close()



