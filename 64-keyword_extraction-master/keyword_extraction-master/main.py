from bidi.algorithm import get_display

from keyword_extraction import Rake
from rake_nlp import Rake_nlp
import arabic_reshaper
from keyword_extraction_rake import Rake_NLTK

text = "Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves."

def main1():

    keyword = Rake(text)
    keyword.get_keywords(10)

def main2():

    keyword = Rake_nlp(text)
    keyword.get_keywords(10)

def main3():

    texte = arabic_reshaper.reshape(text)
    texte = get_display(texte)
    keyword = Rake_NLTK(texte, stopwords=None, punctuations=None, language="english", ranking_metric=0, max_length=100000, min_length=1)

    keyword.get_keywords(10)


if __name__=='__main__':

    main1()

    print("\n")

    main2()

    print("\n")

    main3()
