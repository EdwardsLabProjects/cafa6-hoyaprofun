#!/bin/sh

set -x
if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt 
if [ ! -d .CAFA-evaluator-PK ]; then 
  git clone https://github.com/claradepaolis/CAFA-evaluator-PK.git .CAFA-evaluator-PK
  ln -s .CAFA-evaluator-PK/src/cafaeval 
fi

