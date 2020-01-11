from nltk.tokenize import TweetTokenizer
import re
from nltk.util import ngrams

delimiter = ["atau"]

def token(kata, delimiter = delimiter):
    delimiter = list(set(["dan"]+delimiter))
    pointer = 0
    hasil_token = list()
    deli = list()   
    for i, huruf in enumerate(kata):
        if huruf in delimiter:
            deli.append(i)
    for i in deli:
        h = kata[pointer:i]
        pointer = i+1
        hasil_token.append(h)
    h = kata[pointer:]
    hasil_token.append(h)
    return hasil_token

def token_kata (kata, delimiter = delimiter):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    t_kata = tknzr.tokenize(kata)
    return token(t_kata, delimiter = delimiter)

def ngram (s, n):
    output = ngrams(s, n)
    n=list()
    for i in output:
        n.append("".join(i))
    return n


def ngramku(kata, n=10, delimiter = delimiter):
    token_ = token_kata (kata, delimiter = delimiter)
    # print(token_)
    new_list = list()
    for i in token_:
        if False:
            ix = ngram (i, n)
            new_list.append(i+ix)
        else:
            ix = list()
            for loop in range(2,n+1): #loop n=7 => 1,2,3,
                ix += ngram (i, loop)
            new_list.append(" ".join(i+ix))
    return {"token":" ".join(new_list).split(), "string":" ".join(new_list)}

   
    

# def token_kata (kata, delimiter = delimiter):
#     tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
#     t_kata = tknzr.tokenize(kata)
#     return token(t_kata, delimiter = delimiter)
