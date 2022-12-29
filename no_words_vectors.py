from collections import defaultdict, Counter
from operator import itemgetter
import itertools


def get_sen_and_tagged_sen(filename):
    sentences = []
    sentences_tagged = []
    with open(filename) as file:
        line = True
        while line:
            line = file.readline().split()
            sentence = []
            sentence_tag = []
            for pair in line:
                word, pos = pair.rsplit('/', 1)
                sentence.append(word)
                sentence_tag.append((word, pos))
            sentences.append(sentence)
            sentences_tagged.append(sentence_tag)

    return sentences, sentences_tagged


""" Train """
sentences, sentences_tagged = get_sen_and_tagged_sen("data/ass1-tagger-train")
neighbors_contextual_pos_for_word = defaultdict(lambda: defaultdict(list))
prev_pos_next_pos_counts = defaultdict(lambda: defaultdict(lambda: 0))

for sentence in sentences_tagged:
    for index in range(len(sentence)):
        word, pos = sentence[index]
        previous_word, previous_pos = sentence[index - 1] if index > 0 else (None, None)
        neighbors_contextual_pos_for_word[word][previous_pos].append(pos)
        prev_pos_next_pos_counts[previous_pos][pos] += 1

neighbors_contextual_pos_for_word = {word: dict(neighbors_contextual_pos_for_word[word])
                                     for word in neighbors_contextual_pos_for_word}

""" Validation """
total_words = 0
success = 0
missing = 0
validation_sentences, validation_sentences_tagged = get_sen_and_tagged_sen("data/ass1-tagger-dev")


def get_best_prediction(word, previous_pos):
    if word in neighbors_contextual_pos_for_word:
        try:
            options = neighbors_contextual_pos_for_word[word][previous_pos]
        except:
            options = list(
                itertools.chain.from_iterable(list(neighbors_contextual_pos_for_word[word].values())))
        finally:
            prediction = max(Counter(options).items(), key=itemgetter(1))[0]
    elif word.lower() in neighbors_contextual_pos_for_word:
        try:
            options = neighbors_contextual_pos_for_word[word.lower()][previous_pos]
        except:
            options = list(
                itertools.chain.from_iterable(list(neighbors_contextual_pos_for_word[word.lower()].values())))
        finally:
            prediction = max(Counter(options).items(), key=itemgetter(1))[0]
    else:
        prediction = max(prev_pos_next_pos_counts[previous_pos].items(), key=itemgetter(1))[0]

    return prediction


for sentence in validation_sentences_tagged:
    previous_pos = None
    for index in range(len(sentence)):
        word, actual = sentence[index]
        prediction = get_best_prediction(word, previous_pos)
        if prediction == actual:
            success += 1
        else:
            if word in neighbors_contextual_pos_for_word:
                print("seen", previous_pos, word, prediction, actual,
                      {pos: Counter(options) for pos, options in neighbors_contextual_pos_for_word[word].items()})
            elif word.lower() in neighbors_contextual_pos_for_word:
                print("seen", previous_pos, word, prediction, actual,
                      {pos: Counter(options) for pos, options in neighbors_contextual_pos_for_word[word.lower()].items()})
            else:
                print("unseen", previous_pos, word, prediction, actual)
        previous_pos = prediction
        total_words += 1

print(f"success: {success}")
print(f"total_words: {total_words}")
print(f"success/total_words: {success/total_words}")
print(f"missing: {missing}")


""" Test """

