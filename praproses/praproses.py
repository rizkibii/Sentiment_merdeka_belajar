import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import ngrams as ng
from normalisasi import normalisasi as nrm
import token as tkn
import stemming as stm
import cleansing
import stemming
import seleksi_kata as sk

def praproses2(twt, n=1):
    twt = sk.seleksi(twt)
    twt = cleansing.cleaning(twt)
    twt = nrm.slangword(twt)
    twt = stm.stemming(twt)
    twt = stm.StopWordRemover(twt)
    twt = ng.ngramku(twt, n=n)['string']
    return twt