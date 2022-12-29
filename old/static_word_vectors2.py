import gensim
from collections import defaultdict, Counter
import itertools
from operator import itemgetter
import numpy as np
from numpy.linalg import norm

sentences = []
sentences_tagged = []
validation_sentences = []
validation_sentences_tagged = []
pos_avg_vectors = dict()
pos_count = dict()
words_poss = defaultdict(set)

with open("../data/ass1-tagger-train") as train_file:
    line = True
    while line:
        line = train_file.readline().split()
        sentence = []
        sentence_tag = []
        for index in range(len(line)):
            pair = line[index].split('/')
            last = len(pair) - 1
            words = pair[:last]
            pos = pair[last]
            for word in words:
                word = word.lower()
                sentence.append(word)
                sentence_tag.append((word, pos))
                words_poss[word].add(pos)
        sentences.append(sentence)
        sentences_tagged.append(sentence_tag)

with open("../data/ass1-tagger-dev") as validation_file:
    line = True
    while line:
        line = validation_file.readline().split()
        sentence = []
        sentence_tag = []
        for index in range(len(line)):
            pair = line[index].split('/')
            last = len(pair) - 1
            words = pair[:last]
            pos = pair[last]
            for word in words:
                word = word.lower()
                sentence.append(word)
                sentence_tag.append((word, pos))
        validation_sentences.append(sentence)
        validation_sentences_tagged.append(sentence_tag)

model = gensim.models.Word2Vec(sentences, min_count=1)

for sentence in sentences_tagged:
    for word, pos in sentence:
        if pos in pos_avg_vectors:
            pos_avg_vectors[pos] += model.wv[word].copy()
            pos_count[pos] += 1
        else:
            pos_avg_vectors[pos] = model.wv[word].copy()
            pos_count[pos] = 1

for pos in pos_avg_vectors:
    pos_avg_vectors[pos] /= pos_count[pos]


poss_count = defaultdict(lambda: 0)
neighbors_contextual_pos_for_word = defaultdict(lambda: defaultdict(list))
total_words = 0
success = 0
missing = 0

""" Train """
with open("../data/ass1-tagger-train") as train_file:
    line = True
    while line:
        line = train_file.readline().split()
        for index in range(len(line)):
            pair = line[index].split('/')
            left_pair = line[index - 1].split('/') if index > 0 else [None]
            right_pair = line[index + 1].split('/') if index < len(line) - 1 else [None]
            last, left_last, right_last = len(pair) - 1, len(left_pair) - 1, len(right_pair) - 1
            pos, left_pos, right_pos = pair[last], left_pair[left_last], right_pair[right_last]
            words = pair[:last]
            for word in words:
                # word = word.lower()
                poss_count[pos] += 1
                neighbors_contextual_pos_for_word[word][left_pos].append(pos)


poss_count = dict(poss_count)
neighbors_contextual_pos_for_word = {word: dict(neighbors_contextual_pos_for_word[word])
                                     for word in neighbors_contextual_pos_for_word}

missing_word_options = list(poss_count.keys())
total_appearances = sum(list(poss_count.values()))
weights = tuple([poss_count[pos]/total_appearances for pos in missing_word_options])

""" Validation """
new_data = sentences.copy()
new_data.extend(validation_sentences)
model = gensim.models.Word2Vec(new_data, min_count=1)
with open("../data/ass1-tagger-dev") as train_file:
    line = True
    while line:
        line = train_file.readline().split()
        previous_pos = None
        for index in range(len(line)):
            pair = line[index].split('/')
            last, left_last, right_last = len(pair) - 1, len(left_pair) - 1, len(right_pair) - 1
            actual = pair[last]
            for word in pair[:last]:
                # word = word.lower()
                if word in neighbors_contextual_pos_for_word:
                    word_vec = model.wv[word.lower()]
                    pos_vectors_sim = []
                    try: options = set(neighbors_contextual_pos_for_word[word][previous_pos])
                    except: options = set(itertools.chain.from_iterable(list(neighbors_contextual_pos_for_word[word].values())))
                    for pos in options:
                        pos_vec = pos_avg_vectors[pos]
                        cosine = np.dot(word_vec, pos_vec) / (norm(word_vec) * norm(pos_vec))
                        pos_vectors_sim.append((pos, cosine))
                    closest_pos_vector = max(pos_vectors_sim, key=itemgetter(1))[0]
                    prediction = closest_pos_vector
                else:
                    word_vec = model.wv[word.lower()]
                    pos_vectors_sim = []
                    for pos in pos_avg_vectors:
                        pos_vec = pos_avg_vectors[pos]
                        cosine = np.dot(word_vec, pos_vec) / (norm(word_vec) * norm(pos_vec))
                        pos_vectors_sim.append((pos, cosine))
                    closest_pos_vector = max(pos_vectors_sim, key=itemgetter(1))[0]
                    prediction = closest_pos_vector
                    # prediction = 'JJ'
                    missing += 1
                if prediction == actual:
                    success += 1
                else:
                    if word in neighbors_contextual_pos_for_word:
                        print("seen", word, prediction, actual)
                    else:
                        print("unseen", word, prediction, actual)
                previous_pos = prediction
                total_words += 1

print(f"success: {success}")
print(f"total_words: {total_words}")
print(f"success/total_words: {success/total_words}")
print(f"missing: {missing}")
