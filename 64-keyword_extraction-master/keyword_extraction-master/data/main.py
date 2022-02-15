from translate import Scrapper_Translate
import pandas as pd

labels = []
textes = []

def main_translate():
    try:
        data = pd.read_csv('train.csv', sep=',', encoding='utf-8-sig')
        for text, label in zip(data['text'], data['label']):
            translate = Scrapper_Translate(text =text, browser='Chrome')
            text = translate.Translate()
            if len(text) != 0:
                labels.append(label)
                textes.append(text)
                print(text)
    except:
        dictinnary = {'text': textes, 'label': labels}

        df = pd.DataFrame(dictinnary)

        df.to_csv('data_french_translate.csv', index=False, sep=',', encoding='utf-8-sig')

    dictinnary = {'text': textes, 'label': labels}

    df = pd.DataFrame(dictinnary)

    df.to_csv('data_french_translate.csv', index=False, sep=',', encoding='utf-8-sig')

if __name__=='__main__':
    main_translate()