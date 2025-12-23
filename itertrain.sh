#!/bin/sh

set -x

export PYTHONUNBUFFERED=1

CONFIG="$1"
WORKDIR="$2"
mkdir -p "$WORKDIR"
cp .cache/train_terms_with_anc.tsv $WORKDIR/train_terms_with_anc.0.tsv 
cp "$CONFIG" $WORKDIR/config.ini
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  ./train.py $WORKDIR/config.ini TRAIN_TERMS $WORKDIR/train_terms_with_anc.$i.tsv \
	                         RESULT $WORKDIR/submission.$i.tsv \
				 MODEL_RESULT $WORKDIR/model_submission.$i.tsv \
				 GOA_RESULT $WORKDIR/goa_submission.$i.tsv

  ./eval.py $WORKDIR/config.ini  RESULT $WORKDIR/submission.$i.tsv \
	                         MODEL_RESULT $WORKDIR/model_submission.$i.tsv \
				 EVAL_OUTDIR  $WORKDIR/submissions.$i \
                                 EVAL_GROUNDTRUTH $WORKDIR/.groundtruth.$i.tsv \
                                 EVAL_EXCLUDE $WORKDIR/.exclude.$i.tsv

  j=`expr $i + 1`
  ./update_trainterms.py $WORKDIR/config.ini TRAIN_TERMS_UPDATED $WORKDIR/train_terms_with_anc.$i.tsv \
	                                     RESULT $WORKDIR/submission.$i.tsv \
	                                     > $WORKDIR/train_terms_with_anc.$j.tsv

done

