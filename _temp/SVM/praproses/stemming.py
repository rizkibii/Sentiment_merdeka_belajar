from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(kalimat):
    return stemmer.stem(kalimat)

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory2 = StopWordRemoverFactory()
stopword = factory2.create_stop_word_remover()

def StopWordRemover(kalimat):
    return stopword.remove(kalimat)
 