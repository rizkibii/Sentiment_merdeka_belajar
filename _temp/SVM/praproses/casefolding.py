besar = "ABCDEFGHIJKLMNOPQRSTUVWZYZ"
kecil = "abcdefghijklmnopqrstuvwxyz"
dict_ = dict(zip(besar, kecil))

def casefolding(kalimat):
    kalimat = list(kalimat)
    for ix, huruf in enumerate(kalimat):
        if huruf in dict_:
            kalimat[ix] = dict_[huruf]
    return "".join(kalimat)
