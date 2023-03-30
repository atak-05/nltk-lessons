from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
# import nltk
# nltk.corpus.gutenberg.fileids()
emma =gutenberg.raw('bible-kjv.txt')
tok = sent_tokenize(emma)
print (tok[5:15])



