#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    import re
    #pattern_object = re.compile('regex')
    #search_object = pattern_object.search(string)
    #result_str = search_object.group()


    #Make sure that common items to ignore are part of the stopword list
    stopwords = [""]

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation).strip()

        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")

        escape_seqs = ["\a", "\b", "\f", "\n", "\r", "\t", "\v"]
        for e in escape_seqs:
            text_string = text_string.replace(e, " ")

        words = []
        unstemmed_words = text_string.split(" ")
        
        #print unstemmed_words
        for i,e in enumerate(unstemmed_words):
            if e in stopwords: pass
            else:                
                words.append(stemmer.stem(e.strip()))


    full_string = " ".join(words)

    return full_string

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

