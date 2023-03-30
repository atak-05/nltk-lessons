import nltk
import random
from nltk.corpus import movie_reviews


documents = [(list(movie_reviews.words(fileids=fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)
             ]


random.shuffle(documents)


"""

⁡⁢⁢⁢Bu kod parçası, özellik çıkarımı (feature extraction) yöntemlerinden birini kullanarak, film incelemelerinin özelliklerini (features) belirlemek için kullanılır.
Öncelikle, tüm kelime listesi all_words oluşturulur ve nltk.FreqDist() yöntemi kullanılarak, en sık kullanılan 3000 kelime word_features listesine alınır.
find_features() fonksiyonu, belirli bir inceleme (document) için özellikleri çıkarmak için kullanılır. 
Fonksiyon, inceleme içinde yer alan kelimelerin, word_features listesinde yer alıp almadığını kontrol eder ve özellikleri bu doğrultuda belirler. Sonuç olarak, incelemenin özelliklerini içeren bir sözlük (dictionary) döndürür.
Son olarak, featuresets listesi, her incelemenin özellikleri ile kategorisini içeren bir demet (tuple) olarak oluşturulur. Bu özellik setleri, bir makine öğrenimi modeli eğitilirken kullanılabilir.
⁡
"""





all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets =  [ (find_features(rev),category)for (rev,category) in documents]