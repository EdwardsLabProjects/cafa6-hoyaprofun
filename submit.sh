#!/bin/sh
if [ "$1" = "" -o "$2" = "" ]; then
  echo "Usage: submit.sh submission.tsv \"message\"" 1>&2
  exit 1
fi
./kaggle.sh competitions submit -c cafa-6-protein-function-prediction -f "$1" -m "$2"
