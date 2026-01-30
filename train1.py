#!.venv/bin/python

import sys, glob, os
import steps

print("\n[0/6] Load configuration and initial files...")

CONFIG = steps.configuration(sys.argv[1:],withseed=True)

go = steps.load_ontology(CONFIG)
weights = steps.load_weights(CONFIG)
cumweights = steps.compute_cumweights(CONFIG,weights,go)
test_ids = steps.load_test_ids(CONFIG)
newterms = steps.load_newterms(CONFIG)

# Train proteins only
golabels,protein_to_golabels,golabel_to_proteins = \
                steps.load_train_go_terms(CONFIG,cumweights,restriction=newterms)
train_ids = set(protein_to_golabels)

# Train and test proteins (n, dict)
embed_dim, protein_to_embed = steps.load_protein_embeddings(CONFIG)
ntaxid, protein_to_taxid = steps.load_taxids(CONFIG,train_ids)
ngoaterm, protein_to_goaterm = steps.load_goa_terms(CONFIG,train_ids,golabels)

input_dim, data_dict = steps.prepare_data_loaders2(CONFIG,
                                 goaterm=(ngoaterm, protein_to_goaterm),
                                 taxid=(ntaxid, protein_to_taxid),
                                 embed=(embed_dim, protein_to_embed))

resultfiles = CONFIG['MODEL_RESULT'].replace('.tsv',"-*.tsv")
for fn in glob.glob(resultfiles):
    os.unlink(fn)

data_loader = None
for gl in range(len(golabels)):
    # Prepare training and testing data loaders
    train_loader, val_loader, data_loader = \
        steps.prepare_data_loaders1(CONFIG,train_ids,golabel_to_proteins[gl],
                                    data=(input_dim, data_dict, data_loader))

    model = steps.train_model(CONFIG,train_loader,val_loader)
    model_pred_file1 = CONFIG['MODEL_RESULT'].replace('.tsv',"-%s.tsv"%(gl,))
    steps.predict(CONFIG,model,data_loader,go,[golabels[gl]],filename=model_pred_file1)
    if gl % 10 == 0 and gl != 0:
        steps.merge_preds(CONFIG)

steps.merge_preds(CONFIG)
result_file = CONFIG["RESULT"]
if CONFIG['MERGE_WITH_GOA']:
    goa_pred_file = steps.write_goa_preds(CONFIG)
    steps.combine_preds(CONFIG,result_file)
else:
    shutil.copy(model_pred_file,result_file)






