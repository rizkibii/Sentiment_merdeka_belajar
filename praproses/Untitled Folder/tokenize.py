from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

def tokenisasi(kalimat):
    return tknzr.tokenize(kalimat)

tokenisasi("belajar membaca bahasa indonede,das das")
