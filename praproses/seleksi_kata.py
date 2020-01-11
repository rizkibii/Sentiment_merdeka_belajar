from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
def tokenisasi(kalimat):
    return tknzr.tokenize(kalimat)

def seleksi(kalimat, corpus = ["prabowo", "jokowi", "joko","widodo","sandiaga"]):
    kalimat = tokenisasi(kalimat.lower())
    kalimat = kalimat[2:]
    for ix, kata_ in enumerate(kalimat):
        if kata_ in corpus  or "@" in kata_:
            kalimat[ix] = ""
    return " ".join(kalimat) 