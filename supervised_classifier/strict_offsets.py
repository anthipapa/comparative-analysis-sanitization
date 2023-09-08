import json

with open('data/er_wikitest.json', 'r') as er, open('data/long_wiki_test.json', 'r') as longformer:
	er = json.load(er)
	longfomer = json.load(longformer)


out = {}
for doc_id in longfomer:
	lng = longfomer[doc_id]
	er_detected = er[doc_id]

	strict = [i for i in er_detected if i in lng]
	out[doc_id] = strict



f = open("strict_wiki_test.json", "w")
json.dump(out, f)
f.close()   