#!.venv/bin/python

import sys
import steps

CONFIG = steps.configuration(sys.argv[1:])

result_file = getfile(CONFIG["RESULT"])
df = steps.read_submission(result_file)
train_terms = steps.load_train_terms_ground_truth(CONFIG)
go = steps.load_ontology(CONFIG)

df = df[~df.set_index(["Id","GO term"]).index.isin(train_terms)]
df = df[df.Confidence >= 0.9].nlargest(n=10000,columns='Confidence')

new_train_terms = set(df[["Id","GO term"]].itertuples(index=False))
for id,term in list(new_train_terms):
    t0 = go.get_term(term)
    new_train_terms.update([(id,ti.id) for ti in t0.superclasses(with_self=False)])
train_terms.update(new_train_terms)

ccroot = "GO:0005575"
bproot = "GO:0008150"
mfroot = "GO:0003674"

ccterms = go[ccroot].subclasses().to_set()
bpterms = go[bproot].subclasses().to_set()
mfterms = go[mfroot].subclasses().to_set()

print("EntryId\tterm\taspect")
for t in sorted(train_terms):
    t1 = go.get_term(t[1])
    if t1 in ccterms:
        print("\t".join([t[0],t[1],"C"]))
    elif t1 in bpterms:
        print("\t".join([t[0],t[1],"P"]))
    elif t1 in mfterms:
        print("\t".join([t[0],t[1],"F"]))
