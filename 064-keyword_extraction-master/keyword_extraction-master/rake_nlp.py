from rake_nltk import Rake

class Rake_nlp():
    def __init__(self, text):
        self.text = text
        self.rake = Rake()

    def get_keywords(self, number):
        self.rake.extract_keywords_from_text(self.text)
        #print(self.rake.get_ranked_phrases()[:number])
        print(self.rake.get_ranked_phrases_with_scores()[:number])

