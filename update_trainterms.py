#!.venv/bin/python

import sys
import steps
from util import getfile

CONFIG = steps.configuration(sys.argv[1:])

result_file = getfile(CONFIG["RESULT"],"Results file")
df = steps.read_submission(result_file)
train_terms = steps.load_train_terms_ground_truth(CONFIG)
CONFIG["TRAIN_TERMS"] = CONFIG.get("TRAIN_TERMS_UPDATED","TRAIN_TERMS")
train_terms_updated = steps.load_train_terms_ground_truth(CONFIG)
go = steps.load_ontology(CONFIG)

df1 = df[~df.set_index(["Id","GO term"]).index.isin(train_terms_updated)]
df1 = df1[df1.Confidence >= 0.9]
df2 = df1.nlargest(n=10000,columns='Confidence')
# print(df1.shape)
# print(df2.shape)

df3 = df[~df.set_index(["Id","GO term"]).index.isin(train_terms)]
df3 = df3[df3.set_index(["Id","GO term"]).index.isin(train_terms_updated)]
df3 = df3[df3.Confidence <= 0.1]
df4 = df3.nsmallest(n=1000,columns='Confidence')
# print(df3.shape)
# print(df4.shape)

remove_train_terms_all = set(df3[["Id","GO term"]].itertuples(index=False))
remove_train_terms = set(df4[["Id","GO term"]].itertuples(index=False))

for id,term in list(remove_train_terms_all):
    t0 = go.get_term(term)
    remove_train_terms_all.update([(id,ti.id) for ti in t0.superclasses(with_self=False)])
for id,term in list(remove_train_terms):
    t0 = go.get_term(term)
    remove_train_terms.update([(id,ti.id) for ti in t0.superclasses(with_self=False)])

# print(len(remove_train_terms_all))
# print(len(remove_train_terms))

# print(len(train_terms_updated))
train_terms_updated -= remove_train_terms
# print(len(train_terms_updated))

new_train_terms_all = set(df1[["Id","GO term"]].itertuples(index=False))
new_train_terms = set(df2[["Id","GO term"]].itertuples(index=False))

# print(len(new_train_terms_all))
# print(len(new_train_terms))

for id,term in list(new_train_terms_all):
    t0 = go.get_term(term)
    new_train_terms_all.update([(id,ti.id) for ti in t0.superclasses(with_self=False)])
for id,term in list(new_train_terms):
    t0 = go.get_term(term)
    new_train_terms.update([(id,ti.id) for ti in t0.superclasses(with_self=False)])
print("%d/%d new train terms added to current."%(len(new_train_terms-train_terms_updated),len(new_train_terms_all-train_terms_updated)),file=sys.stderr)
train_terms_updated.update(new_train_terms)
print("%d new train terms over original."%(len(train_terms_updated-train_terms)),file=sys.stderr)

ccroot = "GO:0005575"
bproot = "GO:0008150"
mfroot = "GO:0003674"

ccterms = go[ccroot].subclasses().to_set()
bpterms = go[bproot].subclasses().to_set()
mfterms = go[mfroot].subclasses().to_set()

print("EntryId\tterm\taspect")
for t in sorted(train_terms_updated):
    t1 = go.get_term(t[1])
    if t1 in ccterms:
        print("\t".join([t[0],t[1],"C"]))
    elif t1 in bpterms:
        print("\t".join([t[0],t[1],"P"]))
    elif t1 in mfterms:
        print("\t".join([t[0],t[1],"F"]))
