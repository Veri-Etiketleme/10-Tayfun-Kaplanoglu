import xlwt
from xlrd import open_workbook
from xlutils.copy import copy
import nltk
import nltk
from nltk.corpus import stopwords
nltk.data.path.append('../nltk_data')

list_of_names=[]
name_of_workbook = 'names1.xls'
names = stopwords.words('first_names')
book = open_workbook(name_of_workbook)
sheet = book.sheet_by_index(0)
rb = open_workbook(name_of_workbook)
wb = copy(rb)
w_sheet = wb.get_sheet(0)
for row_val in range(sheet.nrows):
  name = str(sheet.cell_value(row_val, colx=0))
  print name
  if name.upper() in names:
  	pass
  else:
  	if name.upper() not in list_of_names:
  		list_of_names.append(name.upper())
  		w_sheet.write(row_val,2,name.upper())
wb.save(name_of_workbook)  	
  		