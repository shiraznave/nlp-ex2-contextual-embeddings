from collections import defaultdict
from operator import itemgetter
import random
import itertools

poss_count = defaultdict(lambda: 0)
neighbors_contextual_pos_for_word = defaultdict(lambda: defaultdict(list))
total_words = 0
success = 0
missing = 0

""" Train """
with open("data/ass1-tagger-train") as train_file:
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
                word = word.lower()
                poss_count[pos] += 1
                neighbors_contextual_pos_for_word[word][left_pos].append(pos)


poss_count = dict(poss_count)
neighbors_contextual_pos_for_word = {word: dict(neighbors_contextual_pos_for_word[word])
                                     for word in neighbors_contextual_pos_for_word}

missing_word_options = list(poss_count.keys())
total_appearances = sum(list(poss_count.values()))
weights = tuple([poss_count[pos]/total_appearances for pos in missing_word_options])

""" Validation """
with open("data/ass1-tagger-dev") as train_file:
    line = True
    while line:
        line = train_file.readline().split()
        previous_pos = None
        for index in range(len(line)):
            pair = line[index].split('/')
            last, left_last, right_last = len(pair) - 1, len(left_pair) - 1, len(right_pair) - 1
            actual = pair[last]
            for word in pair[:last]:
                word = word.lower()
                total_words += 1
                if word in neighbors_contextual_pos_for_word:
                    try:
                        options = neighbors_contextual_pos_for_word[word][previous_pos]
                    except:
                        options = list(itertools.chain.from_iterable(list(neighbors_contextual_pos_for_word[word].values())))
                    finally:
                        prediction = random.choice(options)
                else:
                    options = poss_count.keys()
                    # prediction = random.choices(missing_word_options, weights=weights, k=1)[0]
                    prediction = 'JJ'
                    missing += 1
                if prediction == actual:
                    success += 1
                previous_pos = prediction

print(f"success: {success}")
print(f"total_words: {total_words}")
print(f"success/total_words: {success/total_words}")
print(f"missing: {missing}")


""" Test """

