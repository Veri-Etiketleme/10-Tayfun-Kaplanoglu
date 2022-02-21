import re, collections
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy
import nltk
from string import punctuation
import enchant,difflib
import nltk
from nltk.corpus import stopwords
import enchant.checker
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter
from nltk.tokenize import RegexpTokenizer
nltk.data.path.append('../nltk_data')
import random
import string

# def words(text): return re.findall('[a-z]+', text.lower()) 

# def train(features):
#     model = collections.defaultdict(lambda: 1)
#     for f in features:
#         model[f] += 1
#     return model

# NWORDS = train(words(file('../data/big.txt').read()))

# alphabet = 'abcdefghijklmnopqrstuvwxyz'

# def edits1(word):
#    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
#    deletes    = [a + b[1:] for a, b in splits if b]
#    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
#    replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
#    inserts    = [a + c + b     for a, b in splits for c in alphabet]
#    return set(deletes + transposes + replaces + inserts)

# def known_edits2(word):
#   return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

# def known(words):
#   return set(w for w in words if w in NWORDS)

# def correct(word):
#     candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
#     return max(candidates, key=NWORDS.get)

corrected_word_dict = {}
d = enchant.Dict("en_US")
# chkr = SpellChecker("en_US",filters=[EmailFilter,URLFilter])
# d = enchant.request_dict("en_US")
name_of_workbook = 'default.xlsx'
names = stopwords.words('first_names')
chat_lingo = stopwords.words('chat_lingo')
technical_terms = stopwords.words('technical_terms')
emoticons = stopwords.words('emoticons')
book = open_workbook(name_of_workbook)
sheet = book.sheet_by_index(0)
rb = open_workbook(name_of_workbook)
wb = copy(rb)
w_sheet = wb.get_sheet(0)
string_url_mapping = {}
for row_val in range(sheet.nrows):
  feedback = sheet.cell_value(row_val, colx=0)
  # feedback = '"neutral","we do a lot of things together in our neighborhood. unorganization at its best"'
  # feedback = feedback.strip(char)
  # feedback = list(feedback)
  # print feedback
  # feedback_split = re.findall(r"[\w']+|[.,!-?;:;)(@]^[a-zA-z]+|[.,!-?;:;)(@]+://[^s].*",feedback)
  # feedback_split = re.split(ur"^[a-zA-z]+|[.,!-?;:;)(@]+://[^s].*|[\u200b\s]+|^.+@[^.].*.[a-z]{2,}$", feedback, flags=re.UNICODE)
  # feedback_split = re.findall(r"[\w']+|[/!?;:;)(#@$]|^[a-zA-z]+://[^s].*",feedback)
  # tokenizer = RegexpTokenizer(r"\w+([.'-]\w+)*")
  # feedback_split = feedback.split()
  urls_and_emails_in_feedback = []
  # urls_and_emails_in_feedback = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', feedback)
  urls_and_emails_in_feedback = re.findall('(?:http[s]*://|www.)[^"\' ]+', feedback)
  # print urls_and_emails_in_feedback
  emails = re.findall(r'[\w\.-]+@[\w\.-]+', feedback)
  if emails:
    urls_and_emails_in_feedback = urls_and_emails_in_feedback+emails
  for url in urls_and_emails_in_feedback:
    #replace each url with a unique placeholder
    random_str = ''.join(random.sample((string.ascii_uppercase+string.digits),6))
    feedback = feedback.replace(url,random_str)

    #also remember which url each random string refers to
    string_url_mapping[random_str] = url
  # feedback_split = re.findall(r"[\w']+|[.,!?;:;)(#@]",feedback)
  feedback_split = re.split("(\W)", feedback)
  # print feedback_split
  new_feedback = ''
  separator = ''
  for index,word in enumerate(feedback_split):
    end = len(word)
    beforeth = word[end-2:end]
    if word.isspace() or word=='':
      new_feedback = new_feedback + word
    elif word=="'":
      continue  
    elif word in ('$','#','@') and index<len(feedback_split)-1:
      new_word = word+feedback_split[index+1]
      feedback_split[index+1] = new_word
      continue
    elif index<len(feedback_split)-2 and feedback_split[index+1]=="'" and feedback_split[index+2] in ('t','T','s','S'):
      new_word = word+feedback_split[index+1]+feedback_split[index+2]
      feedback_split[index+2] = new_word
      continue  
    elif word in string_url_mapping or word[0].isdigit() or (beforeth=="th" and word[:end-2].isdigit()) or (beforeth=="nd" and word[:end-2].isdigit()) or (beforeth=="rd" and word[:end-2].isdigit()) or (beforeth=="st" and word[:end-2].isdigit()) or word.isdigit() or word.endswith(("'s","s'")) or word.startswith(("@","#","$")) or word in emoticons  or word.upper() in names or word.upper() in chat_lingo or word.upper() in technical_terms:
      new_feedback = new_feedback +word
    else:
      if not(d.check(word)) and not(d.check(word.lower())) and not(d.check(word.upper())):
        # corrected_word = correct(word)
        # print word,corrected_word
        # if corrected_word == word:
          # check if the correction already exists
        if word in corrected_word_dict:
          corrected_word = corrected_word_dict[word]
        else:  
          corrected_word = ''
          # best_ratio = 0
          a = d.suggest(word)
          # print a,word
          for b in a:
            # tmp = difflib.SequenceMatcher(None, word, b).ratio()
            # if tmp > best_ratio:
            corrected_word = b
            break
              # best_ratio = tmp
          if corrected_word!='':    
            corrected_word_dict[word] = corrected_word 
        if corrected_word!='':        
          new_feedback = new_feedback + corrected_word
        # print word,corrected_word
      else:
        new_feedback = new_feedback + word
    # if(index<len(feedback_split)-1 and not(feedback_split[index] in punctuation) and feedback_split[index+1] in punctuation):
    #   separator = ''    
    # elif (index<len(feedback_split)-1 and feedback_split[index+1] in punctuation) or (index<len(feedback_split)-1 and (feedback_split[index] in punctuation) and (feedback_split[index+1] in punctuation)):
    #   separator = ''
    # else:
    #   separator = ' '  
  # now replace the random strings with the actual urls
  for key in string_url_mapping:
    new_feedback = new_feedback.replace(key,string_url_mapping[key])
  string_url_mapping.clear()  
  # print new_feedback
  # break 

    # write the new feedback back to the sheet
  w_sheet.write(row_val,3,new_feedback)
wb.save(name_of_workbook)