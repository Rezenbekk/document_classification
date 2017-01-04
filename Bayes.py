## -*- coding: utf-8 -*-

import time
import NormalForm
from collections import defaultdict
from math import log


def train(docs):
    classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for feats, label in docs:
        classes[label] += 1                 # count classes frequencies
        for feat in feats:
            freq[label, feat] += 1          # count features frequencies

    for label, feat in freq:                # normalize features frequencies
        freq[label, feat] /= classes[label]
    for c in classes:                       # normalize classes frequencies
        classes[c] /= len(docs)

    return classes, freq                    # return P(C) and P(O|C)


def classify(classifier, features):
    classes, prob = classifier
    return min(classes.keys(),  # calculate argmin(-log(C|O))
               key=lambda cl: -log(classes[cl]) + \
                                sum(-log(prob.get((cl, feat), 10 ** (-7))) for feat in features))

# Load documents into the memory, stored as Document class objects
train_stemmed_path = "news_train_stemmed.txt"
test_stemmed_path = "news_test_stemmed.txt"
start = time.time()
docs_train_stemmed = NormalForm.fill_doc_list_train(train_stemmed_path)
print("Train set loaded, current time: " + str('%.2f' % (time.time() - start)))
docs_test_stemmed = NormalForm.fill_doc_list_test(test_stemmed_path)
print("Test set loaded, current time: " + str('%.2f' % (time.time() - start)))

features = [(doc.body.split(), doc.category) for doc in docs_train_stemmed]
classifier = train(features)

percentage = 0
index = 0
lng = 15000
file = open("Bayesed.txt", "w", encoding="utf8")
for doc in docs_test_stemmed:
    file.write(classify(classifier, doc.body.split()).name + '\n')
    index += 1
    if percentage - float((index / lng) * 100.0) > 5.0:
        percentage = float((index / lng) * 100.0)
        print("Documents processed: " + str(index) + "  (" + str('%.2f' % percentage) + "%)")
