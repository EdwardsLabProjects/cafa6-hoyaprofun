
import sys
import random
import gc
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pronto import Ontology
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from util import getfile, getconfig

def configuration(args):
    CONFIG = getconfig('config.ini' if len(sys.argv) < 2 else sys.argv[1])["Config"]

    print("Random seed:",CONFIG['RANDOM_SEED'])
    np.random.seed(CONFIG['RANDOM_SEED'])
    torch.manual_seed(CONFIG['RANDOM_SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['RANDOM_SEED'])

    # configure GPU if available....
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIG['device'] = device
    print("Device:",device)

    return CONFIG

def load_weights(CONFIG):
    iafile = getfile("IA.tsv")
    weights = pd.read_csv(iafile, sep='\t', header=None, 
                          names=['term','weight']).set_index("term")
    return weights['weight'].to_dict()

def load_ontology(CONFIG):
    gofile = getfile("go-basic.obo")
    return Ontology(gofile)    

def compute_cumweights(CONFIG,weights,go):
    cumweights = dict()
    for term in weights:
        t0 = go.get_term(term)
        cumweights[term] = sum(weights[ti.id] for ti in t0.superclasses())
    return cumweights

def load_train_ids(CONFIG):
    train_ids_txt = getfile("train_ids.txt")
    return set(open(train_ids_txt).read().split())

def load_test_ids(CONFIG):
    test_ids_txt = getfile("test_ids.txt")
    return set(open(test_ids_txt).read().split())

def load_go_terms(CONFIG,weights,restriction=None):
    print("\n[1/6] Load protein terms to predict...")

    train_terms_with_anc = getfile("train_terms_with_anc.tsv")

    # Make Python dictionary of protein accession to list of GO terms
    train_terms_df = pd.read_csv(train_terms_with_anc, sep='\t', 
                                 header=0, names=['protein', 'term', 'ontology'])
    protein_to_terms = train_terms_df.groupby('protein')['term'].apply(list).to_dict()

    # Compute most frequently used GO terms
    term_counts = Counter()
    for terms in protein_to_terms.values():
        term_counts.update(terms)
    nprotterm = sum(term_counts.values())

    # Top terms by weighted frequency...
    top_terms = [ t[0] for t in sorted(term_counts.items(),
                                       key=lambda t: weights.get(t[0],0.0)*t[1],
                                       reverse=True) ]
    if restriction is not None:
        top_terms = [ t for t in top_terms if t not in restriction ]
    nterms = len(top_terms)
    top_terms = top_terms[:CONFIG['TOP_K_LABELS']]

    term_to_idx = {term: idx for idx, term in enumerate(top_terms)}

    # and restrict protein_to_terms dictionary to only those k terms.
    for protein in protein_to_terms:
        protein_to_terms[protein] = [ term_to_idx[t] for t in protein_to_terms[protein] if t in term_to_idx ]

    ntopprotterm = 0
    for pr in protein_to_terms:
        ntopprotterm += len(protein_to_terms[pr]) 

    print(f">> {len(protein_to_terms)} training proteins, {len(top_terms)} terms selected from {nterms} terms,")
    print(f">> {ntopprotterm} protein-term pairs selected from {nprotterm} protein-term pairs.")

    return top_terms,protein_to_terms

def load_goa_terms(CONFIG,train_ids):

    print("\n[2/6] Load goa terms...")

    # Make Python dictionary of protein accession to list of GOA terms
    goa_uniprot_test_with_anc = getfile("goa_uniprot_test.226_with_anc.tsv")
    goa_terms_df = pd.read_csv(goa_uniprot_test_with_anc, sep='\t', 
                               header=None, usecols=[0,1,2], 
                               names=['protein', 'term', 'evidence'])
    protein_to_goaterms = goa_terms_df.groupby('protein')['term'].apply(list).to_dict()
      
    # Compute most frequently used GOA terms
    term_counts = Counter()
    for pr in protein_to_goaterms:
        if pr in train_ids:
            term_counts.update(protein_to_goaterms[pr])

    nprotgoaterms = sum(term_counts.values())
    ngoaterms = len(term_counts)
    top_goaterms = [ t[0] for t in term_counts.most_common()[:CONFIG['TOP_K_GOATERMS']] ]

    goaterm_to_idx = {term: idx for idx, term in enumerate(top_goaterms)}

    # and restrict protein_to_terms dictionary to only those k terms.
    for protein in protein_to_goaterms:
        protein_to_goaterms[protein] = [goaterm_to_idx[t] for t in protein_to_goaterms[protein] if t in goaterm_to_idx]

    ntopprotgoaterms = 0
    for pr in protein_to_goaterms:
        if pr in train_ids:
            ntopprotgoaterms += len(protein_to_goaterms[pr])

    # record index of each term in top_terms (not ordered?) for 0/1 encoding
    print(f">> {len(protein_to_goaterms)} proteins, {len(top_goaterms)} goa terms selected from train {ngoaterms} goaterms")
    print(f">> {ntopprotgoaterms} protein-goaterm pairs, from {nprotgoaterms} train protein-goaterm pairs.")

    return len(top_goaterms),protein_to_goaterms

def load_taxids(CONFIG,train_ids):
    print("\n[3/6] Load taxids...")

    # all train proteins are found in test...
    testsuperset = getfile("testsuperset.fasta")
    protein_to_taxa = dict()
    for l in open(testsuperset):
        if not l.startswith('>'):
            continue
        pracc,taxid = l[1:].split()
        protein_to_taxa[pracc] = int(taxid)

    taxa_cafa6_test_with_ranks = getfile("taxa_cafa6_test_with_ranks.tsv")
    taxa_to_ranks = pd.read_csv(taxa_cafa6_test_with_ranks, sep='\t')[['taxid','rank','rank_taxid']]
    taxa_to_ranks = taxa_to_ranks.groupby('taxid')['rank_taxid'].apply(list).to_dict()

    # Map taxid to its taxid ranks (species, family, genus, etc.)
    # Compute most frequently used taxids in train data
    taxid_counts = Counter()
    alltaxid = set()
    for pr in list(protein_to_taxa):
        if protein_to_taxa[pr] in taxa_to_ranks:
            protein_to_taxa[pr] = list(taxa_to_ranks[protein_to_taxa[pr]])
            if pr in train_ids:
                taxid_counts.update(protein_to_taxa[pr])
        else:
            del protein_to_taxa[pr]

    ntaxa = len(taxid_counts)
    nprottaxa = sum(taxid_counts.values())

    # Take the most common taxids at any rank...and create index for them
    alltaxid = set([ t[0] for t in taxid_counts.most_common()[:CONFIG['TOP_K_TAXIDS']]])
    taxid_idx = {taxid: idx for idx, taxid in enumerate(alltaxid)}

    for pr in protein_to_taxa:
        protein_to_taxa[pr] = [ taxid_idx[t] for t in protein_to_taxa[pr] if t in alltaxid ]

    ntopprottaxa = 0
    for pr in protein_to_taxa:
        if pr in train_ids:
            ntopprottaxa += len(protein_to_taxa[pr])

    print(f">> {len(protein_to_taxa)} proteins with taxids, {len(alltaxid)} taxids selected from {ntaxa} train taxids")
    print(f">> {ntopprottaxa} protein-taxid pairs selected from {nprottaxa} training protein-taxid pairs.")

    return len(alltaxid),protein_to_taxa

def load_protein_embeddings(CONFIG):

    print("\n[4/6] Loading ProtT5 embeddings...")

    sprot_t5_cafa6_test_ids = getfile("sprot_t5_cafa6_test_ids.npy")
    sprot_t5_cafa6_test_embeds = getfile("sprot_t5_cafa6_test_embeds.npy")
    
    test_ids = np.load(sprot_t5_cafa6_test_ids)
    test_embeds = np.load(sprot_t5_cafa6_test_embeds)

    test_dict = {str(pid): emb for pid, emb in zip(test_ids, test_embeds)}
    embed_dim = test_embeds.shape[1]
    
    print(f">> Protein Embeddings: {test_embeds.shape[0]}, Embedding Dim: {embed_dim}")

    del test_ids, test_embeds
    gc.collect()

    return embed_dim, test_dict

def prepare_data_loaders(CONFIG, **kwargs):

    print("\n[5/6] Preparing training data...")

    ngolabel,protein_to_golabel = kwargs['golabel']
    ntaxid,protein_to_taxid = kwargs['taxid']
    embed_dim,protein_to_embed = kwargs['embed']
    ngoaterm,protein_to_goaterm = kwargs['goaterm']

    valid_train_proteins = set(protein_to_golabel)
    valid_train_proteins &= set(protein_to_goaterm)
    valid_train_proteins &= set(protein_to_taxid)
    valid_train_proteins &= set(protein_to_embed)
    if CONFIG['MAX_TRAIN_SAMPLES'] < len(valid_train_proteins):
        valid_train_proteins = random.sample(valid_train_proteins,k=CONFIG['MAX_TRAIN_SAMPLES'])

    # only for train proteins...
    mlb = MultiLabelBinarizer(classes=range(ngolabel))
    y_labels = [ protein_to_golabel.get(p, []) for p in valid_train_proteins ]
    y_encoded = mlb.fit_transform(y_labels)

    # Apply label smoothing
    if CONFIG['LABEL_SMOOTHING'] > 0:
        y_encoded = y_encoded.astype(float)
        y_encoded = y_encoded * (1 - CONFIG['LABEL_SMOOTHING']) + CONFIG['LABEL_SMOOTHING'] / ngolabel

    valid_proteins = set(protein_to_goaterm)
    valid_proteins &= set(protein_to_taxid)
    valid_proteins &= set(protein_to_embed)
    
    mlb = MultiLabelBinarizer(classes=range(ntaxid))
    tax_labels = [ protein_to_taxid.get(p, []) for p in valid_proteins ]
    tax_encoded = mlb.fit_transform(tax_labels)
    tax_dict = dict(zip(valid_proteins,tax_encoded))

    mlb = MultiLabelBinarizer(classes=range(ngoaterm))
    goaterm_labels = [ protein_to_goaterm.get(p, []) for p in valid_proteins ]
    goaterm_encoded = mlb.fit_transform(goaterm_labels)
    goaterm_dict = dict(zip(valid_proteins,goaterm_encoded))

    data_dict = dict()
    for pr in valid_proteins:
        data_dict[pr] = np.concatenate([protein_to_embed[pr],
                                        tax_dict[pr],
                                        goaterm_dict[pr]
                                       ],axis=0)
    
    inputdim = embed_dim + ntaxid + ngoaterm

    train_proteins, val_proteins, y_train, y_val = train_test_split(
        list(valid_train_proteins), y_encoded, 
        test_size=CONFIG['TRAIN_VAL_SPLIT'], 
        random_state=CONFIG['RANDOM_SEED']
    )
    print(f">> Train: {len(train_proteins)}, Val: {len(val_proteins)}, InputDim: {inputdim}")
    CONFIG['input_dim'] = inputdim
    CONFIG['output_dim'] = ngolabel

    # Make an efficient data-model for training...
    class ProteinDataset(Dataset):
        def __init__(self, proteins, feature_dict, labels):
            self.features = torch.tensor(
                np.array([feature_dict[p] for p in proteins]), 
                dtype=torch.float32
            )
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # This is now just a fast tensor slice, no dictionary hashing needed
            return self.features[idx], self.labels[idx]

    train_dataset = ProteinDataset(train_proteins, data_dict, y_train)
    val_dataset = ProteinDataset(val_proteins, data_dict, y_val)

    del y_train, y_val
    gc.collect()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True,
        pin_memory=(CONFIG['device'] == 'cuda') 
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False,
        pin_memory=(CONFIG['device'] == 'cuda')
    )

    return train_loader, val_loader, data_dict

def train_model(CONFIG,train_loader,val_loader):

    print("\n[6/6] Building model...")

    class ProteinModel(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dims, dropout):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = dim
            self.encoder = nn.Sequential(*layers)
            self.output = nn.Linear(prev_dim, output_dim)
        
        def forward(self, x):
            return self.output(self.encoder(x))

    model = ProteinModel(CONFIG['input_dim'], CONFIG['output_dim'], 
                         CONFIG['HIDDEN_DIMS'], CONFIG['DROPOUT_RATE']).to(CONFIG['device'])
    print(f">> Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ============================================================================
    # TRAINING
    # ============================================================================
    print("\n" + "="*80)
    print(f"TRAINING ({CONFIG['EPOCHS']} EPOCHS WITH LABEL SMOOTHING)")
    print("="*80)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler() if CONFIG['device'] == 'cuda' else None

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            # Non-blocking transfer allows overlap with computation
            X_batch = X_batch.to(CONFIG['device'], non_blocking=True)
            y_batch = y_batch.to(CONFIG['device'], non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            if CONFIG['device'] == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(CONFIG['device'], non_blocking=True)
                y_batch = y_batch.to(CONFIG['device'], non_blocking=True)
                
                if CONFIG['device'] == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                val_batches += 1
                    
        train_loss_avg = epoch_loss/n_batches
        val_loss_avg = val_loss/val_batches
        
        scheduler.step(val_loss_avg)
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            print(f"Epoch {epoch+1}: Train={train_loss_avg:.4f}, Val={val_loss_avg:.4f} * NEW BEST")
            best_weights = model.state_dict()
        else:
            print(f"Epoch {epoch+1}: Train={train_loss_avg:.4f}, Val={val_loss_avg:.4f}")

    model.load_state_dict(best_weights)            
    return model

def predict(CONFIG,model,data_dict,predict_ids,go,golabels,filename="model.tsv"):

    print("\n" + "="*80)
    print("PREDICTIONS (WITH TEMPERATURE SCALING)")
    print("="*80)

    model.eval()
    test_protein_ids = list(set(predict_ids).intersection(set(data_dict.keys())))

    n_predictions = 0
    with open(filename, 'w', newline='') as f:
        with torch.no_grad():
            for start in tqdm(range(0, len(test_protein_ids), CONFIG['PREDICT_BATCH_SIZE']), desc="Predicting", ascii=True):
                batch_ids = test_protein_ids[start:start + CONFIG['PREDICT_BATCH_SIZE']]
                X_batch = torch.FloatTensor(numpy.array([data_dict[p] for p in batch_ids])).to(CONFIG['device'])
                
                if CONFIG['device'] == 'cuda':
                    with torch.amp.autocast('cuda'):
                        logits = model(X_batch)
                else:
                    logits = model(X_batch)
                
                # Apply temperature scaling
                outputs = torch.sigmoid(logits / CONFIG['TEMPERATURE']).cpu().numpy()
                
                for i, pid in enumerate(batch_ids):
                    probs = outputs[i]
                    top_indices = np.argsort(probs)[::-1][:CONFIG['MAX_PREDS_PER_PROTEIN']]
                    confident_indices = [idx for idx in top_indices if probs[idx] > CONFIG['MIN_CONFIDENCE']]

                    term_probs = {}
                    for idx in confident_indices:
                        term_probs[golabels[idx]] = probs[idx]

                    for term in list(term_probs):
                        t0 = go.get_term(term)
                        for ti in t0.superclasses(with_self=False):
                            if term_probs[term] > term_probs.get(ti.id,0.0):
                                term_probs[ti.id] = term_probs[term]

                    for term,prob in term_probs.items():
                        if prob <= 0.0:
                            continue
                        line = f"{pid}\t{term}\t{min(prob, 0.999):.3f}\n"
                        f.write(line)
                        n_predictions += 1
                
                del X_batch, outputs, logits
                if start % 1000 == 0:
                    gc.collect()

    print(f"✓ Generated {n_predictions:,} predictions")
    return filename

    # from pylab import *
    # dl_df = pd.read_csv('temp_dl.tsv', sep='\t', header=None, names=['Id', 'GO term', 'Confidence'])
    # hist(dl_df.Confidence,50)
    # show()
    # del model
    # gc.collect()



