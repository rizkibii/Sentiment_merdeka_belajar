import pandas as pd
import os
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
def tokenisasi(kalimat):
    return tknzr.tokenize(kalimat)

dir_path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(dir_path+"/corpus.csv", sep=';' )
kata  = df['kata'].tolist()
# print(kata)
replace  = df['replace'].tolist()
dict_ = dict(zip(kata, replace))

def slangword(kalimat):
    # print(kalimat)
    try:
        kalimat = tokenisasi(kalimat)
    except:
        kalimat = kalimat.split()
    for ix, kata_ in enumerate(kalimat):
        if kata_ in dict_:
            kalimat[ix] = dict_[kata_]
    return " ".join(kalimat)
# print(dict_)