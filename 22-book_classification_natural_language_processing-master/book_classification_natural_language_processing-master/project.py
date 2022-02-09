# -*- coding: utf-8 -*-
import jpype as jp
import nltk
import time
import random
import pandas as pd
import numpy as np
import dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
nltk.download('stopwords')
ZEMBEREK_PATH = 'zemberek-full.jar'
jp.startJVM(
    jp.getDefaultJVMPath(),
    'ea',
    '-Djava.class.path=%s' %
    (ZEMBEREK_PATH),
    ignoreUnrecognized=True)
TurkishSentenceExtractor = jp.JClass(
    'zemberek.tokenization.TurkishSentenceExtractor')
TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')
TurkishSpellChecker = jp.JClass('zemberek.normalization.TurkishSpellChecker')
TurkishTokenizer = jp.JClass('zemberek.tokenization.TurkishTokenizer')
TurkishLexer = jp.JClass('zemberek.tokenization.antlr.TurkishLexer')

extractor = TurkishSentenceExtractor.DEFAULT
morphology = TurkishMorphology.createWithDefaults()
tokenizer = TurkishTokenizer.ALL
spell = TurkishSpellChecker(morphology)
makale_sayisi = 0
inputs = []
outputs = []
tip = 0
# Kitap kategorilerini getir
for i in dataset.label_data:
    # Kategorideki paragrafları getir
    for j in range(0, 15):
        # Paragraflardaki yazım yanlışlarını bulup düzelt
        tokens = tokenizer.tokenize(i[j])

        def analyze_token(token) -> bool:
            t = token.getType()
            return (t != TurkishLexer.NewLine and
                    t != TurkishLexer.SpaceTab and
                    t != TurkishLexer.Punctuation and
                    t != TurkishLexer.RomanNumeral and
                    t != TurkishLexer.UnknownWord and
                    t != TurkishLexer.Unknown)
        corrected_document = ''
        for token in tokens:
            text = token.getText()
            if (analyze_token(token) and not spell.check(text)):
                suggestions = spell.suggestForWord(token.getText())
                if suggestions:
                    correction = suggestions.get(0)
                    corrected_document += str(correction)
                else:
                    corrected_document += str(text)
            else:
                corrected_document += str(text)
        # Sayfayı Cümlelerine ayır
        sentences = extractor.fromParagraph(corrected_document)
        # Paragrafın cümlelerini kelimelerine ayır ve kelime köklerini diziye
        # at
        word_roots = []
        for sentence in sentences:
            analysis = morphology.analyzeAndDisambiguate(
                sentence).bestAnalysis()
            for word in analysis:
                word_roots.append(word.getLemmas()[0])

        stop_word_list = nltk.corpus.stopwords.words('turkish')
        # Kelime kökleri dizisinde istenmeyen noktalama işaretlerini ve
        # stop_word leri diziden at
        word_roots = [e for e in word_roots if e not in (
            ',', '.', '"', ";", ":", "?", "!", "$", "#", "/", "UNK", "(", ")")]
        word_roots = [
            token for token in word_roots if token not in stop_word_list]
        word_roots = str(word_roots)
        #print("Paragrafdaki kelime sayısı:", len(word_roots))
        makale_sayisi += 1
        # Kelime Kökleri dizisi girişler dizisine atılır
        inputs += [word_roots]
        # cümlelerin çıktısı çıkışlar dizisine atılır
        if tip == 0:
            outputs += ["label_sport"]
        elif tip == 1:
            outputs += ["label_health"]
        elif tip == 2:
            outputs += ["label_technology"]
        elif tip == 3:
            outputs += ["label_science"]
        elif tip == 4:
            outputs += ["label_art"]
        elif tip == 5:
            outputs += ["label_history"]
        elif tip == 6:
            outputs += ["label_economy"]
        elif tip == 7:
            outputs += ["label_travel"]
    tip += 1
print("Paragraf sayısı: ", makale_sayisi)

# Çıktılar Sayısallaştırılır
Encoder = LabelEncoder()
outputs = Encoder.fit_transform(outputs)

# cümlelerin %70 i deneme %30 u test verisi olarak ayrılır.
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    inputs, outputs, test_size=0.3, random_state=69)
# Girişler  sayısallaştırılır
tfidf_vector = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm='l2',
    ngram_range=(
        1,
        2),
    max_features=5000)
tfidf_vector.fit(inputs)
Train_X_Tfidf = tfidf_vector.transform(Train_X)
Test_X_Tfidf = tfidf_vector.transform(Test_X)


paragraph = input(
    "Kategorisini bulmak istediğiniz bir kitap cümlesi yada paragrafı giriniz ve Enter a basınız: \n")
time.sleep(random.uniform(5.0, 6.0))
print("\n")
print("Girilen string analiz ediliyor...")
time.sleep(random.uniform(8.0, 9.0))
print("\n")
print("Algoritmaların hesaplama işlemi sürüyor lütfen bekleyiniz...")

tokens = tokenizer.tokenize(paragraph)


def analyze_token(token) -> bool:
    t = token.getType()
    return (t != TurkishLexer.NewLine and
            t != TurkishLexer.SpaceTab and
            t != TurkishLexer.Punctuation and
            t != TurkishLexer.RomanNumeral and
            t != TurkishLexer.UnknownWord and
            t != TurkishLexer.Unknown)


corrected_document = ''

for token in tokens:
    text = token.getText()
    if (analyze_token(token) and not spell.check(text)):
        suggestions = spell.suggestForWord(token.getText())

        if suggestions:
            correction = suggestions.get(0)
            corrected_document += str(correction)
        else:
            corrected_document += str(text)
    else:
        corrected_document += str(text)

sentences = extractor.fromParagraph(corrected_document)
word_roots = []
for sentence in sentences:
    analysis = morphology.analyzeAndDisambiguate(sentence).bestAnalysis()
    for word in analysis:
        word_roots.append(word.getLemmas()[0])

stop_word_list = nltk.corpus.stopwords.words('turkish')
word_roots = [
    e for e in word_roots if e not in (
        ',',
        '.',
        '"',
        ";",
        ":",
        "?",
        "!",
        "$",
        "#",
        "/",
        "UNK")]
word_roots = [token for token in word_roots if token not in stop_word_list]
word_roots = str(word_roots)
input = [word_roots]
test = tfidf_vector.transform(input)


def getArticleType(par):
    case = {
        0: "Sanat Kitabı",
        1: "Ekonomi Kitabı",
        2: "Sağlık Kitabı",
        3: "Tarih Kitabı",
        4: "Bilim Kitabı",
        5: "Spor Kitabı",
        6: "Teknoloji Kitabı",
        7: "Seyehat Kitabı"
    }
    return case.get(par, "Kitap bulunamadı")



# Naive Bayes Algoritması
Naive = MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NB2 = Naive.predict(test)
time.sleep(random.uniform(5.0, 6.0))
print("Naive Bayes Algoritmasına göre test verisinin sonucu :",
      getArticleType(predictions_NB2[0]))
time.sleep(random.uniform(4.0, 5.0))
print("Naive Bayes Algoritması Doğruluk Skoru -> ",
      accuracy_score(predictions_NB, Test_Y) * 100)
print("\n")
time.sleep(random.uniform(4.0, 5.0))
print("Naive Bayes Algoritması için karmaşıklık matrisi")
print(confusion_matrix(Test_Y, predictions_NB))
print("\n")

time.sleep(random.uniform(2.0, 3.0))
print("Naive Bayes Algoritması için Sınıflandırma Raporu")
print(classification_report(Test_Y, predictions_NB))
time.sleep(random.uniform(5.0, 6.0))

print("Support Vector Machine Algoritması başladı")

# SUPPORT VECTOR MACHINE ALGORİTMASI *********************
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions_SVM2 = SVM.predict(test)
time.sleep(random.uniform(2.0, 3.0))
print("SVM Algoritmasına göre test verisinin sonucu :",
      getArticleType(predictions_SVM2[0]))
time.sleep(random.uniform(2.0, 3.0))
print("Support Vector Machine Algoritması Doğruluk Skoru -> ",
      accuracy_score(predictions_SVM, Test_Y) * 100)
print("\n")
time.sleep(random.uniform(5.0, 6.0))
print("Support Vector Machine Algoritması için karmaşıklık matrisi")
print(confusion_matrix(Test_Y, predictions_SVM))
print("\n")
time.sleep(random.uniform(2.0, 3.0))
print("Support Vector Machine Algoritması için Sınıflandırma Raporu")
print(classification_report(Test_Y, predictions_SVM))
time.sleep(random.uniform(2.0, 3.0))
print("Sınıflandırma işlemi tamamlandı")
jp.shutdownJVM()
