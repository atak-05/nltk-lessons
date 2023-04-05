import pickle
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from  sklearn.svm import NuSVC, SVC, LinearSVC
from nltk.classify import ClassifierI
from statistics import mode



"""
⁡⁢⁢⁢The expression "combining algorithms with a vote" refers to the process of using more than one algorithm in a forecasting or decision-making process and combining the results of these algorithms using the voting mechanism.
This method is frequently used in the fields of machine learning and artificial intelligence, and it provides more accurate results by analyzing data from different perspectives of multiple algorithms.
⁡
"""

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len((votes))
        return conf
                


"""
⁡⁢⁢⁢​‌‍‌Naive Bayes ​sınıflandırıcısı, makine öğrenmesinde sınıflandırma problemlerini çözmek için kullanılan bir algoritmadır. 
Bu algoritma, verileri istatistiksel bir yöntemle sınıflandırır. Özellikle doğal dil işlemede, spam filtreleme, duygu analizi gibi sınıflandırma problemlerinde sıklıkla kullanılmaktadır.
Naive Bayes sınıflandırıcısı, Bayes teoremi temel alınarak oluşturulur. 
Bayes teoremi, koşullu olasılık hesaplamalarında kullanılan bir matematiksel formüldür. Naive Bayes algoritması ise bu teoreme dayalı bir makine öğrenmesi algoritmasıdır.
Naive Bayes sınıflandırıcısı, öğrenme aşamasında eğitim verilerini kullanarak her özelliğin sınıfa ait olasılıklarını hesaplar. 
Sınıflandırma yaparken, test verilerinin sınıfına ait olasılıkları hesaplar ve en yüksek olasılığa sahip sınıfı seçer.
Naive Bayes sınıflandırıcısı, özellikle küçük boyutlu veri setlerinde etkili sonuçlar verir ve hızlı çalışır. Ancak, "naive" yani "saf" olarak adlandırılmasının nedeni, modelin özellikler arasındaki bağımlılıkları göz ardı etmesidir. Bu nedenle, bağımsız olmayan özelliklere sahip veri setleri için daha az uygun olabilir.
"""





documents =[(list(movie_reviews.words(fileids=fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category) 
            ]
# random.shuffle(documents)



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

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets =  [ (find_features(rev),category)for (rev,category) in documents]

""" 
⁡⁢⁢⁡⁢⁢⁢training_set, ilk 1900 belgeyi içeren bir özellik-kategori çiftleri listesidir
ve testing_set, geri kalan belgeleri içeren bir liste.
Sınıflandırıcı, nltk.NaiveBayesClassifier.train() yöntemi kullanılarak eğitilir ve doğruluğu test edilir. 
Ayrıca, show_most_informative_features() yöntemi kullanılarak enformatik özellikler yazdırılır.⁡
"""
#Positive data example:
training_set = featuresets[:1900]
testing_set = featuresets[1900:]


#Negative data example:
training_set = featuresets[100:]
testing_set = featuresets[:100]

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
print ("Original Naive Bayes Algo accuracy percent :  ",(nltk.classify.accuracy(classifier,testing_set))*100 )
classifier.show_most_informative_features(15)
print("--------------MNB_classifie------------------")
"""
⁡⁢⁢⁢Bu kod, Scikit-learn kütüphanesinin​‌‍‌ 𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀​ sınıflandırıcısını kullanarak bir sınıflandırıcı eğitmeyi ve test etmeyi gösterir.
İlk olarak, SklearnClassifier sınıfı kullanılarak, NLTK sınıflandırma modeli Scikit-learn sınıflandırma modeline dönüştürülür. 
Daha sonra, MultinomialNB() sınıflandırıcısı kullanılarak eğitim setindeki belgelerin sınıflandırılması yapılır.
Eğitim tamamlandıktan sonra, test setindeki belgelerin doğruluğu hesaplanır ve ekrana yazdırılır.
Bu yöntem, Naive Bayes sınıflandırıcısı yerine Scikit-learn'ün MultinomialNB sınıflandırıcısının kullanılmasına olanak tanır. MultinomialNB sınıflandırıcısı, Naive Bayes sınıflandırıcısının bir varyasyonudur 
ve özellikle metin sınıflandırma problemleri için uygun bir seçenektir.⁡
"""
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print ("MNB_classifier accuracy percent :  ",(nltk.classify.accuracy(MNB_classifier,testing_set))*100 )


# print("-----------BernoulliNB---------------------")
""" 
⁡⁢⁢⁢​‌‍‌GaussianNB (Gaussian Naive Bayes)​ sınıflandırıcısı, Bayes teoremine dayalı bir olasılık modeli kullanarak sınıflandırma yapar.
Bu sınıflandırıcı, özellik vektörlerinin öznitelik değerlerinin normal dağılım (Gaussian distribution) ile modellendiği varsayımına dayanır.
Bu varsayım altında, sınıfların yoğunlukları belirli bir çizgi boyunca normal olarak dağıtılmıştır.
Bu sınıflandırıcı, 𝘀𝗮𝘆ı𝘀𝗮𝗹 𝘃𝗲𝗿𝗶𝗹𝗲𝗿𝗶𝗻 𝘀ı𝗻ı𝗳𝗹𝗮𝗻𝗱ı𝗿ı𝗹𝗺𝗮𝘀ı 𝗶ç𝗶𝗻 𝘂𝘆𝗴𝘂𝗻𝗱𝘂𝗿.
Aşağıdaki kod bloğunda, GaussianNB sınıflandırıcısı, SklearnClassifier sınıfı aracılığıyla NLTK sınıflandırma yapısına dahil edilir. Ardından, train yöntemi kullanılarak eğitilir ve test seti üzerindeki doğruluğu hesaplanır.⁡
"""
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print ("GaussianNB_classifier accuracy percent :  ",(nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100 )

print("----------- BernoulliNB,---------------------")
"""
​‌‍‌⁡⁢⁢⁢BernoulliNB,​ Bernoulli Dağılımı'nı varsayan bir Naive Bayes sınıflandırıcısıdır.
Bu, her özellik için 0 veya 1 olabilecek bir Bernoulli değişkeni varsayarak çalışır.
Eğitim verileri özelliklerin bir varlığını veya yokluğunu belirtir. 
Bu durumda, özellik setimiz "belirli bir kelimenin varlığı" veya "belirli bir kelimenin yokluğu" gibi olabilir.
Bu nedenle, BernoulliNB, bir belgeye dahil edilen kelime sayısına değil,
yalnızca belgeye dahil edilen veya dahil edilmeyen kelimelerin sayısına dayanır.
BernoulliNB, ö𝘇𝗲𝗹𝗹𝗶𝗸𝗹𝗲 𝗯𝗶𝗿 𝗯𝗲𝗹𝗴𝗲𝗱𝗲𝗸𝗶 𝗸𝗲𝗹𝗶𝗺𝗲 𝘀𝗮𝘆ı𝘀ı ç𝗼𝗸 𝗱𝗲ğ𝗶ş𝗸𝗲𝗻 𝗼𝗹𝗱𝘂ğ𝘂𝗻𝗱𝗮 𝗸𝘂𝗹𝗹𝗮𝗻ış𝗹ı𝗱ı𝗿.⁡

"""
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print ("BernoulliNB_classifier accuracy percent :  ",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100 )


# LogisticRegression, SGDClassifier
# NuSVC, SVC, LinearSVC

print("----------- LogisticRegression_classifier,---------------------")
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print ("LogisticRegression_classifier accuracy percent :  ",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100 )


print("----------- SGDClassifier,---------------------")
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print ("SGDClassifier accuracy percent :  ",(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100 )


print("----------- NuSVC_classifier,---------------------")
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print ("NuSVC_classifier accuracy percent :  ",(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100 )


# print("----------- SVC_classifier,---------------------")
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print ("SVC_classifier accuracy percent :  ",(nltk.classify.accuracy(SVC_classifier,testing_set))*100 )


print("----------- LinearSVC_classifier,---------------------")
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print ("LinearSVC_classifier accuracy percent :  ",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100 )







"""   ⁡⁢⁢⁢********************************** ​‌‍‌Voted Classifier​ *****************************⁡ """

voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier,
                                  SGDClassifier_classifier,
                                  LogisticRegression_classifier,
                                  BernoulliNB_classifier)


print ("voted_classifier accuracy percent :  ",(nltk.classify.accuracy(voted_classifier,testing_set))*100 )

# print("Classification : ", voted_classifier.classify(testing_set[0][0]),"Confidence %:", voted_classifier.confidence(testing_set[0][0])*100 )
# print("Classification : ", voted_classifier.classify(testing_set[1][0]),"Confidence %:", voted_classifier.confidence(testing_set[1][0])*100 )
# print("Classification : ", voted_classifier.classify(testing_set[2][0]),"Confidence %:", voted_classifier.confidence(testing_set[2][0])*100 )
# print("Classification : ", voted_classifier.classify(testing_set[3][0]),"Confidence %:", voted_classifier.confidence(testing_set[3][0])*100 )
# print("Classification : ", voted_classifier.classify(testing_set[4][0]),"Confidence %:", voted_classifier.confidence(testing_set[4][0])*100 )
# print("Classification : ", voted_classifier.classify(testing_set[5][0]),"Confidence %:", voted_classifier.confidence(testing_set[5][0])*100 )