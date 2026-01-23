#!/bin/sh
if [ "$1" = "" -o "$2" = "" ]; then
  echo "Usage: submit.sh submission.tsv \"message\"" 1>&2
  exit 1
fi
TMP_DIR=$(mktemp -d) || exit 1
cp "$1" "$TMP_DIR"/submission.tsv
trap 'rm -rf "$TMP_DIR"' EXIT
./kaggle.sh competitions submit -c cafa-6-protein-function-prediction -f "$TMP_DIR/submission.tsv" -m "$2"
