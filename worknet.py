from nltk.corpus import wordnet


syns = wordnet.synsets("program")

#synset
print (syns[0].name)


#just the word 
print (syns[0].lemmas()[0].name())

#definition
print(syns[0].definition())

#examples
print(syns[0].examples())



sysnonyms = []
antonyms  = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        print("l: ", l)
        sysnonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
# print(set(sysnonyms))
# print(set(antonyms))

# //* Daha sonra w1.wup_similarity(w2) yöntemi çağrılır. Bu, synset'ler arasındaki Wu-Palmer benzerliği ölçer. 
# */ Wu-Palmer benzerliği, iki kelimenin semantik olarak ne kadar yakın olduğunu ölçmek için kullanılır.


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
similarity_score = w1.wup_similarity(w2)
print(similarity_score)



w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
similarity_score = w1.wup_similarity(w2)
print(similarity_score)


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
similarity_score = w1.wup_similarity(w2)
print(similarity_score)