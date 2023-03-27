import nltk

# nltk.download()

#tokenzing - word tokenizers.... sentence tokenizers
# lexicon and corporas
# corpara - body of text.. ex: medical journals, presidential speeches, English Language
# investor-speak.. regular english-speak..
# investor speak 'bull' = someone who is positive about the market
# english-speak 'bull' = scary animal you dont want running .. you


from nltk.tokenize import sent_tokenize, word_tokenize

example_text = """Hello Mr. Smith, how are you doing today?
The weather is great and Python is awesome! 
The sky is pinkish-blue.
You should not eat cardboard.
"""
print(sent_tokenize(example_text)) #Sentence tokenizers
print("----------------------")

for i in word_tokenize(example_text): #Word tokenizers
    print(i)
