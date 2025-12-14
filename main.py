#!.venv/bin/python

import sys
import steps

print("\n[0/6] Load configuration and initial files...")

CONFIG = steps.configuration(sys.argv)

go = steps.load_ontology(CONFIG)
weights = steps.load_weights(CONFIG)
cumweights = steps.compute_cumweights(CONFIG,weights,go)
train_ids = steps.load_train_ids(CONFIG)
test_ids = steps.load_test_ids(CONFIG)

# Train proteins only (n, dict)
golabels, protein_to_golabel = steps.load_go_terms(CONFIG,cumweights)

# Train and test proteins (n, dict)
ngoaterm, protein_to_goaterm = steps.load_goa_terms(CONFIG,train_ids)
ntaxid, protein_to_taxid = steps.load_taxids(CONFIG,train_ids)
embed_dim, protein_to_embed = steps.load_protein_embeddings(CONFIG)

# Prepare training data loaders and testing data dictionary
train_loader, val_loader, data_dict = \
    steps.prepare_data_loaders(CONFIG, 
                               golabel=(len(golabels), protein_to_golabel),
                               goaterm=(ngoaterm, protein_to_goaterm),
                               taxid=(ntaxid, protein_to_taxid),
                               embed=(embed_dim, protein_to_embed))

model = steps.train_model(CONFIG,train_loader,val_loader)

model_pred_file = steps.predict(CONFIG,model,data_dict,test_ids,go,golabels)
