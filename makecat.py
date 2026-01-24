#!.venv/bin/python

import hashlib
import os, os.path, sys

for f in sys.argv[1:]:
    sf = f.split('/')
    dropbox = sf[-2]
    filename = sf[-1]
    md5 = str(hashlib.md5(open(f,'rb').read()).hexdigest().lower())
    size = str(os.path.getsize(f))
    print("\t".join([dropbox,filename,md5,size]))


