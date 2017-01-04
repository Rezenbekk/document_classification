# -*- coding: utf-8 -*-

import Document as Dc
import re
import Stemmer
from io import open


def stem_file_train(source_path, out_path):
    stop_words = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
                  'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только',
                  'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
                  'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас',
                  'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они',
                  'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
                  'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот',
                  'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой',
                  'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
                  'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти',
                  'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем',
                  'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им',
                  'более', 'всегда', 'конечно', 'всю', 'между', 'что', 'это', 'так', 'вот', 'быть', 'как', 'в',
                  '—', 'к', 'на']
    alphabet = '[^абвгдежзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz\t\n ]'
    regex = re.compile(alphabet)

    docs_by_lines = []
    with open(source_path, 'r', encoding='utf8') as infile:
        for line in infile:
            line = line.lower().replace("ё", "е").replace('.', '. ').replace(';', '; ').replace(',', ', ')
            doc_parts = line.split('\t')
            doc_parts[1] = regex.sub('', doc_parts[1])
            doc_parts[2] = regex.sub('', doc_parts[2])
            doc_parts[1] = (' '.join([word for word in doc_parts[1].split() if word not in stop_words])).strip()
            doc_parts[2] = (' '.join([word for word in doc_parts[2].split() if word not in stop_words])).strip()
            line = '\t'.join(doc_parts)
            docs_by_lines.append(line)

    stemmer = Stemmer.Stemmer('russian')
    stemmer.maxCacheSize = 0

    file = open(out_path, 'w', encoding='utf8')

    for index, line in enumerate(docs_by_lines):
        doc = line.split('\t')

        file.write(doc[0]+'\t')

        words = doc[1].split(' ')
        for word in words:
            word = stemmer.stemWord(word)
            file.write(word+' ')
        file.write('\t')

        words = doc[2].split(' ')
        for word in words:
            word = stemmer.stemWord(word)
            file.write(word + ' ')
        file.write('\n')


def stem_file_test(source_path, out_path):
    stop_words = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
                  'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только',
                  'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
                  'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас',
                  'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они',
                  'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
                  'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот',
                  'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой',
                  'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
                  'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти',
                  'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем',
                  'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им',
                  'более', 'всегда', 'конечно', 'всю', 'между', 'что', 'это', 'так', 'вот', 'быть', 'как', 'в',
                  '—', 'к', 'на']
    dictionary = '[^абвгдежзийклмнопрстуфхцчшщъыьэюя \t\n]'
    regex = re.compile(dictionary)

    docs_by_lines = []
    with open(source_path, 'r', encoding='utf8') as infile:
        for line in infile:
            line = line.lower().replace("ё", "е").replace('.', '. ').replace(';', '; ').replace(',', ', ')
            line = regex.sub('', line)
            doc_parts = line.split('\t')
            doc_parts[0] = regex.sub('', doc_parts[0])
            doc_parts[1] = regex.sub('', doc_parts[1])
            doc_parts[0] = ' '.join([word for word in doc_parts[0].split() if word not in stop_words])
            doc_parts[1] = ' '.join([word for word in doc_parts[1].split() if word not in stop_words])
            line = '\t'.join(doc_parts)
            docs_by_lines.append(line)

    stemmer = Stemmer.Stemmer('russian')
    stemmer.maxCacheSize = 0

    file = open(out_path, 'w', encoding='utf8')

    for index, line in enumerate(docs_by_lines):
        doc = line.split('\t')

        words = doc[0].split(' ')
        for word in words:
            word = stemmer.stemWord(word)
            file.write(word+' ')
        file.write('\t')

        words = doc[1].split(' ')
        for word in words:
            word = stemmer.stemWord(word)
            file.write(word + ' ')
        file.write('\n')


def fill_doc_list_train(file_path, amount=0):
    documents_list = []
    docs_splitted = []

    with open(file_path, 'r', encoding='utf8') as infile:
        for line in infile:
            docs_splitted.append(line)
            if not amount:
                if (len(docs_splitted)) == amount:
                    break

    for doc in docs_splitted:
        single_doc = doc.split("\t")
        new_doc = Dc.Document(single_doc[0], single_doc[1], single_doc[2])
        documents_list.append(new_doc)
    # list of Document class objects
    return documents_list


def fill_doc_list_test(file_path, amount=0):
    documents_list = []
    docs_splitted = []

    with open(file_path, 'r', encoding='utf8') as infile:
        for line in infile:
            line = line.lower()
            docs_splitted.append(line)
            if not amount:
                if (len(docs_splitted)) == amount:
                    break

    for doc in docs_splitted:
        single_doc = doc.split("\t")
        new_doc = Dc.Document('', single_doc[0], single_doc[1])
        documents_list.append(new_doc)
    # list of Document class objects
    return documents_list


print("Custom file NormalForm.py has been imported.")

# For testing purposes

# start = time.time()
#
# train_path = "news_train.txt"
# test_path = "news_test.txt"
# train_stemmed_path = "news_train_stemmed.txt"
# test_stemmed_path = "news_test_stemmed.txt"

# Stem the files and store them for future use
# stem_file_train(train_path, train_stemmed_path)
# end = time.time()
# print("Finished stemming " + train_path + ", current time: " + str('%.3f' % (end - start)))
# stem_file_test(test_path, test_stemmed_path)
# end = time.time()
# print("Finished stemming " + test_path + ", current time: " + str('%.3f' % (end - start)))

# Load documents into the memory, stored as Document class objects
# docs_train_stemmed = fill_doc_list_train(train_stemmed_path)
# docs_test_stemmed = fill_doc_list_test(test_stemmed_path)

# end = time.time()
# print("Total time: " + str('%.3f' % (end - start)))
# print(docs_train_stemmed[567].body)
# print(docs_test_stemmed[568].body)
