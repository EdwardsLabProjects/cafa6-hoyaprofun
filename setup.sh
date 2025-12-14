#!/bin/sh

if [ ! -d .venv ]; then
  "${1:-python3}" -m venv .venv
fi
if [ ! -d .CAFA-evaluator-PK ]; then 
  git clone https://github.com/claradepaolis/CAFA-evaluator-PK.git .CAFA-evaluator-PK
  ln -s .CAFA-evaluator-PK/src/cafaeval 
fi
.venv/bin/python -m pip install -r requirements.txt 

