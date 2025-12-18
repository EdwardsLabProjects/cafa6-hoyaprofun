
import sys
import os
import os.path
import csv
import random
import gc
import shutil
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pronto import Ontology
import numpy as np
import pandas as pd
import pylab
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from cafaeval.evaluation import cafa_eval, write_results
import logging

from util import getfile, getconfig

def configuration(args):
    config_filename = ('config.ini' if len(args) < 1 else args[0])
    config = getconfig(config_filename)
    CONFIG = config["Config"]
    CONFIG.update(config["Files"])
    for i in range(1,len(args),2):
        CONFIG[args[i]] = eval(args[i+1])

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

def load_train_terms_ground_truth(CONFIG,ancestors=True,asset=True):
    if ancestors:
        fn = getfile(CONFIG["TRAIN_TERMS"])
    else:
        fn = getfile(CONFIG["TRAIN_TERMS_NOANC"])
    df = pd.read_csv(fn, sep='\t', header=0, usecols=[0,1], names=['protein', 'term'])
    if not asset:
        return df
    return set(df.itertuples(index=False, name=None))

def load_ground_truth(CONFIG,ancestors=True,asset=True):
    if ancestors:
        fn = getfile(CONFIG["GROUND_TRUTH"])
        df = pd.read_csv(fn, sep='\t', header=None, usecols=[0,1,4], names=['protein', 'term','exp'])
    else:
        fn = getfile(CONFIG["GROUND_TRUTH_NOANC"])
        df = pd.read_csv(fn, sep='\t', header=None, usecols=[0,1,6], names=['protein', 'term','exp'])
    df = df[df.exp == 'EXP']
    df = df[['protein','term']]
    if not asset:
        return df
    return set(df.itertuples(index=False, name=None))

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
    nprot = len(protein_to_terms)

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
    goa_uniprot_test_with_anc = getfile(CONFIG["GOA_INPUT"])
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
        def __init__(self, proteins, feature_dict, labels=None):
            proteins = list(proteins)
            self.features = torch.tensor(
                np.array([feature_dict[p] for p in proteins]), 
                dtype=torch.float32
            )
            self.proteins = proteins
            self.len = len(proteins)
            self.labels = None
            if labels is not None:
                self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            # This is now just a fast tensor slice, no dictionary hashing needed
            if self.labels is not None:
                return self.features[idx], self.labels[idx]
            return self.proteins[idx], self.features[idx]

    train_dataset = ProteinDataset(train_proteins, data_dict, y_train)
    val_dataset = ProteinDataset(val_proteins, data_dict, y_val)
    valid_dataset = ProteinDataset(valid_proteins, data_dict)

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
        batch_size=CONFIG['PREDICT_BATCH_SIZE'], 
        shuffle=False,
        pin_memory=(CONFIG['device'] == 'cuda')
    )

    data_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG['PREDICT_BATCH_SIZE'],
        shuffle=False,
        pin_memory=(CONFIG['device'] == 'cuda')
    )

    return train_loader, val_loader, data_loader

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
            best_weights_from = (epoch+1,train_loss_avg,best_val_loss)
        else:
            print(f"Epoch {epoch+1}: Train={train_loss_avg:.4f}, Val={val_loss_avg:.4f}")

    print(f"Setting model to best weights - Epoch {best_weights_from[0]}: Train={best_weights_from[1]:.4f} Val={best_weights_from[0]:.4f}")
    model.load_state_dict(best_weights)            
    return model

def predict(CONFIG,model,data_loader,go,golabels,filename=None):

    if filename is None:
        filename = CONFIG['MODEL_RESULT']

    print("\n" + "="*80)
    print("PREDICTIONS (WITH TEMPERATURE SCALING)")
    print("="*80)

    model.eval()

    n_predictions = 0
    with open(filename, 'w', newline='') as f:
        with torch.no_grad():
            for batch_ids, X_batch in tqdm(data_loader, desc="Predicting", total=len(data_loader), ascii=True):
                X_batch = X_batch.to(CONFIG['device'], non_blocking=True)
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
                
                del X_batch, outputs, logits, batch_ids

    print(f">> Generated {n_predictions:,} predictions")
    return filename

def write_goa_preds(CONFIG,filename=None):
    if filename is None:
        filename = CONFIG["GOA_RESULT"]
    goa_pred_file = getfile(CONFIG["GOA_MERGE"])
    chunks = []
    for chunk in pd.read_csv(goa_pred_file,
                             header=None, sep='\t', 
                             usecols = [0,1],
                             names=["Id","GO term"],
                             chunksize=1000000):
        chunk['Confidence'] = 1.0
        chunks.append(chunk)
        if len(chunks) >= 10:
            df = pd.concat(chunks, ignore_index=True)
            chunks = [df]
            gc.collect()

    df = pd.concat(chunks, ignore_index=True)

    df.to_csv(filename, sep='\t', header=False, index=False,
              quoting=csv.QUOTE_NONE, escapechar='\\', lineterminator='\n',
              float_format="%.3f")

    return filename

def read_submission(filename,**kwargs):
    if 'header' not in kwargs:
        kwargs['header'] = None
    if 'names' not in kwargs:
        kwargs['names'] = ["Id","GO term","Confidence"]
    chunks = []
    for chunk in pd.read_csv(filename, sep='\t', chunksize=1000000, **kwargs):
        chunks.append(chunk)
        if len(chunks) >= 10:
            df = pd.concat(chunks, ignore_index=True)
            chunks = [df]
            gc.collect()
    return pd.concat(chunks, ignore_index=True)

def combine_preds(CONFIG,model_pred_file,goa_pred_file,pred_file=None):

    if pred_file is None:
        pred_file = CONFIG["RESULT"]

    goa_df = read_submission(CONFIG["GOA_RESULT"])
    goa_df['GOA_Confidence'] = goa_df['Confidence']
    goa_df['Confidence'] = goa_df['Confidence'] * CONFIG['GOA_WEIGHT']

    model_df = read_submission(CONFIG["MODEL_RESULT"])
    model_df["Model_Confidence"] = model_df["Confidence"]
    model_df["Confidence"] = model_df["Confidence"] * CONFIG["MODEL_WEIGHT"]

    print(f">> Model={len(model_df):,}, GOA={len(goa_df):,}")
  
    ensemble_df = pd.concat([model_df, goa_df], ignore_index=True)
    del model_df, goa_df
    gc.collect()
    
    ensemble_df = ensemble_df.groupby(['Id', 'GO term'], as_index=False).sum()
    
    bothpositive = ((ensemble_df['GOA_Confidence'] > 0.0) & (ensemble_df['Model_Confidence'] > 0.0))
    ensemble_df.loc[bothpositive,'Confidence'] += CONFIG['CONSENSUS_BOOST']
    ensemble_df['Confidence'] = ensemble_df['Confidence'].clip(lower=0.0,upper=1.0)
        
    ensemble_df = ensemble_df[ensemble_df['Confidence'] > 0.0]
    
    ensemble_df = ensemble_df[['Id', 'GO term', 'Confidence']]
    ensemble_df = ensemble_df.sort_values(['Id', 'Confidence'], ascending=[True, False])
    ensemble_df = ensemble_df.groupby('Id').head(1500)
  
    ensemble_df.to_csv('submission.tsv', sep='\t', header=False, index=False,
                       quoting=csv.QUOTE_NONE, escapechar='\\', lineterminator='\n',
                       float_format="%.3f")
    print(f">> Saved {len(ensemble_df):,} predictions")

def write_submission_plot(CONFIG,*filenames,outfile=None):
    if outfile is None:
        if len(filenames) == 1:
            base = filenames[0].rsplit('.',1)[0]
            outfile = base + "_confidence.png"
        else:
            outfile = "confidence.png"
    pylab.figure()
    for f in filenames:
        df = read_submission(f)
        base = f.rsplit('.',1)[0]
        pylab.plot(sorted(df['Confidence'],reverse=True),'.',label=base)
    pylab.legend(loc="upper right")
    pylab.savefig(outfile)

def compute_results(base,gt,df,ignore=None):
    retval = []
    scale=100
    for thr in set(map(int,df.Confidence*scale)):
        pred = set(df.loc[df.Confidence*scale>=thr,["Id","GO term"]].itertuples(index=False, name=None))
        if ignore is not None:
            pred.difference_update(ignore)
        tp = len(pred&gt)
        fp = len(pred)-tp
        fn = len(gt)-tp
        if (tp + fp) == 0 or (tp + fn) == 0:
            continue
        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        if (prec+recall)==0:
            continue
        f1 = 2*(prec*recall)/(prec + recall)
        print(base,"%.2f"%(thr/scale,),tp,fp,fn,"%.3f"%(100*tp/(tp+fp),),"%.3f"%(100*tp/(tp+fn),),"%.3f"%(f1,))
        retval.append((base,thr,tp,fp,fn,prec,recall,f1))
    return retval

def write_precall_plot(CONFIG,ground_truth,*filenames,ignore=None,outfile=None):
    if outfile is None:
        if len(filenames) == 1:
            base = filenames[0].rsplit('.',1)[0]
            outfile = base + "_precall.png"
        else:
            outfile = "precall.png"
    if ignore is not None:
        ground_truth1 = ground_truth.difference(ignore)
    else:
        ground_truth1 = ground_truth
    pylab.figure()
    for f in filenames:
        df = read_submission(f)
        base = f.rsplit('.',1)[0]
        pr = compute_results(base,ground_truth1,df,ignore=ignore)
        x = [ t[6] for t in pr]
        y = [ t[5] for t in pr]
        x = [x[0]] + x + [0]
        y = [0] + y + [y[-1]]
        pylab.plot(x,y,label=base)
    pylab.legend()
    pylab.savefig(outfile)

def run_cafa6_eval(CONFIG,*filenames,outdir='submissions'):
    gt = load_ground_truth(CONFIG,ancestors=False) 
    obo_file = getfile("go-basic.obo")
    ia_file =  getfile("IA.tsv")
    train_gt = load_train_terms_ground_truth(CONFIG,ancestors=False) 

    if os.path.isdir(outdir):                                                                      
        shutil.rmtree(outdir)                                                                      
    os.makedirs(outdir)

    gt_file = 'ground_truth.tsv'
    with open(gt_file,'wt') as wh:
        for pracc,goacc in sorted(gt-train_gt): # remove all train terms
            print("\t".join([pracc,goacc]),file=wh)

    exclude_file = 'exclude.tsv' # exclude all train terms
    with open(exclude_file,'wt') as wh:
        for pracc,goacc in sorted(train_gt):
            print("\t".join([pracc,goacc]),file=wh)

    logging.basicConfig()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName('INFO'))
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    root_handler = root_logger.handlers[0]
    root_handler.setFormatter(log_formatter)
    
    for f in filenames:                                                                                   
        shutil.copy(f,outdir+'/'+f)
    th_step = 0.01
    df, dfs_best = cafa_eval(obo_file=obo_file, pred_dir=outdir, 
                             gt_file=gt_file,
                             ia=ia_file, exclude=exclude_file, 
                             max_terms=1500, th_step=th_step, 
                             n_cpu=1)
    write_results(df, dfs_best, out_dir=outdir, th_step=th_step)

def cafa6_plots(CONFIG,evaldir='submissions'):
    # Input files
    df_file = evaldir+"/"+"evaluation_all.tsv"
    out_folder = evaldir

    # Set to None if you don't want to use it. Results will not be grouped/filtered by team
    names_file = None

    # Cumulate the last column of the cols variable, e.g. "pr" --> precision, so that the curves are monotonic as in CAFA
    cumulate = True

    # Add extreme points to the precision-recall curves (0, 1) and (1, 0)
    add_extreme_points = True

    # Methods with coverage below this threshold will not be plotted
    coverage_threshold = 0.3

    # Select a metric
    # metric, cols = ('f', ['rc', 'pr'])
    metric, cols =  ('f_w', ['rc_w', 'pr_w'])
    # metric, cols =  ('f_micro', ['rc_micro', 'pr_micro'])
    # metric, cols =  ('f_micro_w', ['rc_micro_w', 'pr_micro_w'])
    # metric, cols = ('s_w', ['ru_w', 'mi_w'])

    # Map column names to full names (for axis labels)
    axis_title_dict = {'pr': 'Precision', 'rc': 'Recall', 'f': 'F-score', 'pr_w': 'Weighted Precision', 'rc_w': 'Weighted Recall', 'f_w': 'Weighted F-score', 'mi': 'Misinformation (Unweighted)', 'ru': 'Remaining Uncertainty (Unweighted)', 'mi_w': 'Misinformation', 'ru_w': 'Remaining Uncertainty', 's': 'S-score', 'pr_micro': 'Precision (Micro)', 'rc_micro': 'Recall (Micro)', 'f_micro': 'F-score (Micro)', 'pr_micro_w': 'Weighted Precision (Micro)', 'rc_micro_w': 'Weighted Recall (Micro)', 'f_micro_w': 'Weighted F-score (Micro)'}

    # Map ontology namespaces to full names (for plot titles)
    ontology_dict = {'biological_process': 'BPO', 'molecular_function': 'MFO', 'cellular_component': 'CCO'}

    df = pd.read_csv(df_file, sep="\t")

    # Set method information (optional)
    if names_file is None:
        df['group'] = df['filename']
        df['label'] = df['filename']
        df['is_baseline'] = False
    else:
        methods = pd.read_csv(names_file, delim_whitespace=True, header=0)
        df = pd.merge(df, methods, on='filename', how='left')
        df['group'].fillna(df['filename'], inplace=True)
        df['label'].fillna(df['filename'], inplace=True)
        if 'is_baseline' not in df:
            df['is_baseline'] = False
        else:
            df['is_baseline'].fillna(False, inplace=True)
        # print(methods)
    df = df.drop(columns='filename').set_index(['group', 'label', 'ns', 'tau'])

    # Filter by coverage
    df = df[df['cov'] >= coverage_threshold]

    # Assign colors based on group
    cmap = pylab.get_cmap('tab20')
    df['colors'] = df.index.get_level_values('group')
    df['colors'] = pd.factorize(df['colors'])[0]
    df['colors'] = df['colors'].apply(lambda x: cmap.colors[x % len(cmap.colors)])

    # Identify the best methods and thresholds
    index_best = df.groupby(level=['group', 'ns'])[metric].idxmax() if metric in ['f', 'f_w', 'f_micro', 'f_micro_w'] else df.groupby(['group', 'ns'])[metric].idxmin()
    
    # Filter the dataframe for the best methods
    df_methods = df.reset_index('tau').loc[[ele[:-1] for ele in index_best], ['tau', 'cov', 'colors'] + cols + [metric]].sort_index()

    # Makes the curves monotonic. Cumulative max on the last column of the cols variable, e.g. "pr" --> precision
    if cumulate:
        if metric in ['f', 'f_w', 'f_micro', 'f_micro_w']:
            df_methods[cols[-1]] = df_methods.groupby(level=['label', 'ns'])[cols[-1]].cummax()
        else:
            df_methods[cols[-1]] = df_methods.groupby(level=['label', 'ns'])[cols[-1]].cummin()


    # Save to file
    df_methods.drop(columns=['colors']).to_csv('{}/fig_{}.tsv'.format(out_folder, metric), float_format="%.3f", sep="\t")

    # Add first last points to precision and recall curves to improve APS calculation
    def add_points(df_):
        df_ = pd.concat([df_.iloc[0:1], df_])
        df_.iloc[0, df_.columns.get_indexer(['tau', cols[0], cols[1]])] = [0, 1, 0]  # tau, rc, pr
        df_ = pd.concat([df_, df_.iloc[-1:]])
        df_.iloc[-1, df_.columns.get_indexer(['tau', cols[0], cols[1]])] = [1.1, 0, 1]
        return df_

    if metric.startswith('f') and add_extreme_points:
        df_methods = df_methods.reset_index().groupby(['group', 'label', 'ns'], as_index=False).apply(add_points).set_index(['group', 'label', 'ns'])

    # Filter the dataframe for the best method and threshold
    df_best = df.loc[index_best, ['cov', 'colors'] + cols + [metric]]

    # Calculate average precision score 
    if metric.startswith('f'):
        df_best['aps'] = df_methods.groupby(level=['group', 'label', 'ns'])[[cols[0], cols[1]]].apply(lambda x: (x[cols[0]].diff(-1).shift(1) * x[cols[1]]).sum())

    # Calculate the max coverage across all thresholds
    df_best['max_cov'] = df_methods.groupby(level=['group', 'label', 'ns'])['cov'].max()

    # Set a label column for the plot legend
    df_best['label'] = df_best.index.get_level_values('label')
    if 'aps' not in df_best.columns:
        df_best['label'] = df_best.agg(lambda x: f"{x['label']} ({metric.upper()}={x[metric]:.3f} C={x['max_cov']:.3f})", axis=1)
    else:
        df_best['label'] = df_best.agg(lambda x: f"{x['label']} ({metric.upper()}={x[metric]:.3f} APS={x['aps']:.3f} C={x['max_cov']:.3f})", axis=1)

    # Generate the figures
    pylab.rcParams.update({'font.size': 22, 'legend.fontsize': 18})

    # F-score contour lines
    x = np.arange(0.01, 1, 0.01)
    y = np.arange(0.01, 1, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = 2 * X * Y / (X + Y)

    for ns, df_g in df_best.groupby(level='ns'):
        fig, ax = pylab.subplots(figsize=(15, 15))

        # Contour lines. At the moment they are provided only for the F-score
        if metric.startswith('f'):
            CS = ax.contour(X, Y, Z, np.arange(0.1, 1.0, 0.1), colors='gray')
            ax.clabel(CS, inline=True) #, fontsize=10)

        # Iterate methods
        for i, (index, row) in enumerate(df_g.sort_values(by=[metric, 'max_cov'], ascending=[False if metric.startswith('f') else True, False]).iterrows()):
            data = df_methods.loc[index[:-1]]
            
            # Precision-recall or mi-ru curves
            ax.plot(data[cols[0]], data[cols[1]], color=row['colors'], label=row['label'], lw=2, zorder=500-i)
            
            # F-max or S-min dots
            ax.plot(row[cols[0]], row[cols[1]], color=row['colors'], marker='o', markersize=12, mfc='none', zorder=1000-i)
            ax.plot(row[cols[0]], row[cols[1]], color=row['colors'], marker='o', markersize=6, zorder=1000-i)

        # Set axes limit
        if metric.startswith('f'):
            pylab.xlim(0, 1)
            pylab.ylim(0, 1)
    
        # pylab.xlim(0, max(1, df_best.loc[:,:,ns,:][cols[0]].max()))
        # pylab.ylim(0, max(1, df_best.loc[:,:,ns,:][cols[1]].max()))

        # Set titles
        ax.set_title(ontology_dict.get(ns, ns), pad=20)
        ax.set_xlabel(axis_title_dict[cols[0]], labelpad=20)
        ax.set_ylabel(axis_title_dict[cols[1]], labelpad=20)
        
        # Legend
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg = ax.legend(markerscale=6)
        for legobj in leg.get_lines():
            legobj.set_linewidth(10.0)

        # Save figure on disk
        pylab.savefig("{}/fig_{}_{}.png".format(out_folder, metric, ns), bbox_inches='tight', dpi=300, transparent=True)
        # pylab.clf()

    replace = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}
    df_group = df_group.droplevel(level=1)                                                            
    df_group.reset_index(inplace=True)                                                                
    fws = dict(df_group[["ns","f_w"]].itertuples(index=False))                                        
    print(group,"F_w",end=" ")                                                                
    for k,v in sorted(fws.items()):                                                                   
        k = replace.get(k)                                                                            
        print(f"{k}: {v}",end=" ")                                                                    
    print("Overall:","%.3f"%(sum(fws.values())/3),)