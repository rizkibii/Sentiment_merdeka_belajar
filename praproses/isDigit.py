import re
def remove_digit(kalimat):
    kalimat = kalimat.split()
    for ix, kata in enumerate(kalimat):
        if kata.isdigit() == True:
            kalimat[ix]=""
    kalimat = " ".join(kalimat)
    return kalimat.strip()
# print(remove_digit("12321 das 2dasd"))
