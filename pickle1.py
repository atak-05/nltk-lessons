import pickle
import nltk
import random
from nltk.corpus import movie_reviews


"""
⁡⁢⁢⁢​‌‍‌Naive Bayes ​sınıflandırıcısı, makine öğrenmesinde sınıflandırma problemlerini çözmek için kullanılan bir algoritmadır. 
Bu algoritma, verileri istatistiksel bir yöntemle sınıflandırır. Özellikle doğal dil işlemede, spam filtreleme, duygu analizi gibi sınıflandırma problemlerinde sıklıkla kullanılmaktadır.
Naive Bayes sınıflandırıcısı, Bayes teoremi temel alınarak oluşturulur. 
Bayes teoremi, koşullu olasılık hesaplamalarında kullanılan bir matematiksel formüldür. Naive Bayes algoritması ise bu teoreme dayalı bir makine öğrenmesi algoritmasıdır.
Naive Bayes sınıflandırıcısı, öğrenme aşamasında eğitim verilerini kullanarak her özelliğin sınıfa ait olasılıklarını hesaplar. 
Sınıflandırma yaparken, test verilerinin sınıfına ait olasılıkları hesaplar ve en yüksek olasılığa sahip sınıfı seçer.
Naive Bayes sınıflandırıcısı, özellikle küçük boyutlu veri setlerinde etkili sonuçlar verir ve hızlı çalışır. Ancak, "naive" yani "saf" olarak adlandırılmasının nedeni, modelin özellikler arasındaki bağımlılıkları göz ardı etmesidir. Bu nedenle, bağımsız olmayan özelliklere sahip veri setleri için daha az uygun olabilir.
⁡
"""


documents = [(list(movie_reviews.words(fileids=fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)
             ]


random.shuffle(documents)



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
""" 
⁡⁢⁢⁡⁢⁢⁢training_set, ilk 1900 belgeyi içeren bir özellik-kategori çiftleri listesidir
ve testing_set, geri kalan belgeleri içeren bir liste.
Sınıflandırıcı, nltk.NaiveBayesClassifier.train() yöntemi kullanılarak eğitilir ve doğruluğu test edilir. 
Ayrıca, show_most_informative_features() yöntemi kullanılarak enformatik özellikler yazdırılır.⁡
"""

training_set = featuresets[:1900]

testing_set = featuresets[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)


"""
​‌‍⁡⁢⁢⁢‌naivebayes.pickle ​dosyasını okuyor ve içindeki eğitilmiş sınıflandırıcıyı classifier değişkenine yüklüyor. 
Ardından, bu sınıflandırıcıyı test veri seti (testing_set) üzerinde değerlendiriyor ve doğruluk oranını ekrana yazdırıyor.
Son olarak, sınıflandırıcının en bilgilendirici özelliklerini ekrana yazdırıyor.

Bu kod, daha önce eğitilmiş bir sınıflandırıcının yüklenmesini gösteriyor. 
Bu şekilde, sınıflandırıcıyı yeniden eğitmeden önce kaydedilen eğitilmiş bir modele erişebilirsiniz.
pickle modülü, Python nesnelerini serileştirmek ve deserialize etmek için kullanılır.
Bu örnekte, pickle kullanılarak eğitilmiş sınıflandırıcı bir dosyaya kaydedilmiş ve daha sonra bu dosyadan tekrar yüklenebilmiştir.⁡
"""
classifier_f =open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
print ("Naive Bayes Algo accuracy percent :  ",(nltk.classify.accuracy(classifier,testing_set))*100 )
classifier.show_most_informative_features(15)

# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
