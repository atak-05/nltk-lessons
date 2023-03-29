from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better", pos="n"))

print(lemmatizer.lemmatize("best", pos="a"))

print(lemmatizer.lemmatize("run"))

print(lemmatizer.lemmatize("run", pos="v"))
