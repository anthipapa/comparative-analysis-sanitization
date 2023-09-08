import json

with open('data/er_tabtest.json', 'r') as er, open('data/long_tab_test.json', 'r') as longformer, open('data/gold/echr_test.json', 'r') as gold:
	er = json.load(er)
	longfomer = json.load(longformer)
	gold = json.load(gold)


out = {}
for doc in gold:
	doc_id = doc['doc_id']
	lng = longfomer[doc_id]
	er_detected = er[doc_id]

	out[doc_id] = [i for i in er_detected if i in lng]

	for tpl1 in lng:
		for tpl2 in er_detected:
			if tpl1[0] == tpl2[0] and tpl1[1] == tpl2[1]: ## exact match
				if tpl1 not in out[doc_id]:
					out[doc_id].append(tpl1)
			if tpl1[0] == tpl2[0]: ## start is the same
				if tpl1 not in out[doc_id]:
					out[doc_id].append(tpl1)
			elif tpl1[1] == tpl2[1]: ## end is the same
				if tpl1 not in out[doc_id]:
					out[doc_id].append(tpl1)





f = open("partial_tab_test.json", "w")
json.dump(out, f)
f.close()   
