import urllib.request, hashlib, os, os.path, csv, random, sys

base = "https://edwardslab.bmcb.georgetown.edu/~nedwards/dropbox/"
catalog = None
cache = ".cache"

def download(dropbox,filename,size,hash):
    tofile = cache + "/" + filename
    os.makedirs(cache,exist_ok=True)
    if not os.path.exists(tofile):
        print("Downloading %s... "%(filename,),end="",file=sys.stderr)
        sys.stdout.flush()
        urllib.request.urlretrieve(base+"/"+dropbox+"/"+filename,tofile)
        md5 = hashlib.md5(open(tofile,'rb').read()).hexdigest().lower()
        assert (size == os.path.getsize(tofile)) and (md5 == hash)
        print("done.",file=sys.stderr)
    else:
        md5 = hashlib.md5(open(tofile,'rb').read()).hexdigest().lower()
        if (size != os.path.getsize(tofile)) or (md5 !=  hash):
            os.unlink(tofile)
            return download(dropbox,filename,size,hash)
        else:
            print("Using cached file %s. "%(filename,),file=sys.stderr)
    return tofile

def file_catalog():
    catalog = {}
    for r in csv.DictReader(open('catalog.tsv'),dialect='excel-tab'):
        r['size'] = int(r['size'])
        catalog[r['filename']] = dict(r.items())
    return catalog

def getfile(filename):
    if os.path.exists(filename):
        print("Using local file %s. "%(filename,),file=sys.stderr)
        return filename
    assert filename in catalog
    data = catalog[filename]
    return download(data['dropbox'],filename,data['size'],data['md5'])
    
catalog = file_catalog()

from configparser import ConfigParser

def getconfig(filename):
    retval = dict()
    cp = ConfigParser()
    cp.optionxform = str
    cp.read(filename)
    for sec in cp.sections():
        retval[sec] = dict()
        for k,v in cp.items(sec):
            retval[sec][k] = eval(v)
    return retval

