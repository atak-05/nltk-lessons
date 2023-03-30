import nltk
import random
from nltk.corpus import movie_reviews



"""
⁡⁢⁢⁢Bu kod, NLTK'nin içinde bulunan movie_reviews corpus'undaki belgeleri okuyarak bir belge listesi oluşturur. 
Her belge, bir film eleştirisini ve eleştirinin pozitif mi yoksa negatif mi olduğunu gösteren bir etiketi içerir. 
Bu belgeler daha sonra karıştırılır.

movie_reviews corpus'u, 2000 film eleştirisinden oluşan bir koleksiyondur. 
Her eleştiri, pozitif veya negatif olarak etiketlenmiştir.
Bu kod, her eleştiriyi bir belge olarak ele alır ve belgeleri rastgele karıştırarak, 
doğrulama ve test setleri oluşturmak için kullanılabilir.

documents[1] çıktısı, belgeler listesindeki ikinci belge ve onun etiketidir.
⁡
"""
documents = [(list(movie_reviews.words(fileids=fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)
             ]


random.shuffle(documents)
# print(documents[0])


"""
​‌‍‌‍‍⁡⁢⁣⁢nltk.FreqDist()​ ⁡⁢⁢⁢bir sınıftır ve frekans dağılımı hesaplamak için kullanılır.
Farklı veri türlerindeki elemanların sayısını sayabilir. ⁡
​‌‍‌⁡⁢⁣⁢
most_common()⁡​ ⁡⁢⁢⁢yöntemi, sıklıkları büyükten küçüğe doğru sıralar ve en yaygın olanları belirtilen sayıda listeler.⁡ 

"""


"""
⁡⁢⁢⁢document = []
 for category in movie_reviews.categories():
     for fileid in movie_reviews.fileids(category):
         document.append(list(movie_reviews.words(fileid)), category)⁡
"""


all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))    

print(all_words["stupid"]) #örneğin burda 253 kez bu kelime geçmiş olduğu sonucu görürüz!!

"""
⁡⁢⁢⁢Bu kod bloğu, NLTK'in "movie_reviews" corpus'undan tüm kelimeleri alır ve bir frekans dağılımı nesnesine ekler. 
Bu nesne, her kelimenin kaç kez geçtiğini sayar.
Daha sonra, en sık kullanılan 15 kelimeyi yazdırmak için "most_common()" yöntemi kullanılır.⁡
"""