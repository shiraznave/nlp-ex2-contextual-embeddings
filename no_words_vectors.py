from collections import defaultdict
from operator import itemgetter
import random
import itertools

pairs_count = defaultdict(lambda: 0)
poss_count = defaultdict(lambda: 0)
words_count = defaultdict(lambda: 0)
words_pos_counts = defaultdict(lambda: defaultdict(lambda: 0))
most_common_pos_for_word = {}
neighbors_contextual_pos_for_word = defaultdict(lambda: defaultdict(list))

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
                pairs_count[tuple([word, pos])] += 1
                poss_count[pos] += 1
                words_pos_counts[word][pos] += 1
                neighbors_contextual_pos_for_word[word][left_pos].append(pos)


pairs_counts = dict(sorted(pairs_count.items(), key=itemgetter(1), reverse=False))
poss_count = dict(poss_count)
words_count = dict(words_count)
words_pos_counts = {word: dict(words_pos_counts[word]) for word in words_pos_counts}
neighbors_contextual_pos_for_word = {word: dict(neighbors_contextual_pos_for_word[word])
                                     for word in neighbors_contextual_pos_for_word}

# print(pairs_counts)
for pair in pairs_count:
    if len(pair) > 2:
        print(pair)

# print(poss_count)
# print(words_pos_counts)

non_single_pos = 0
for word, poss in words_pos_counts.items():
    most_common_pos_for_word[word] = max(poss.items(), key=itemgetter(1))
    if len(poss) > 1:
        # print(word, poss)
        # print(word, most_common_pos_for_word[word])
        non_single_pos += 1

# print(most_common_pos_for_word)

# word = 'are'
# print(most_common_pos_for_word[word])
# print(words_pos_counts[word])

total_words = 0
success = 0
missing = 0

# model = neighbors_contextual_pos_for_word
# for word in model:
#     # if len(model[word]) > 1:
#         # print(word, model[word])
#     for context in model[word]:
#         if len(model[word][context]) > 1:
#             print(word, context, model[word][context])

missing_word_options = list(poss_count.keys())
total_appearances = sum(list(poss_count.values()))
weights = tuple([poss_count[pos]/total_appearances for pos in missing_word_options])

# with open("data/ass1-tagger-dev") as train_file:
#     line = True
#     while line:
#         line = train_file.readline().split()
#         for pair in line:
#             split_pair = pair.split('/')
#             last_index = len(split_pair) - 1
#             actual = split_pair[last_index]
#             for word in split_pair[:last_index]:
#                 word = word.lower()
#                 total_words += 1
#                 prediction = most_common_pos_for_word[word] if word in most_common_pos_for_word \
#                     else random.choices(missing_word_options, weights=weights, k=1)[0]
#                 if word not in most_common_pos_for_word:
#                     missing += 1
#                 if prediction == actual:
#                     success += 1


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
print(f"non_single_pos: {non_single_pos}")