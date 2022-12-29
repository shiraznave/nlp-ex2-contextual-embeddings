import gensim
from collections import defaultdict, Counter
from operator import itemgetter
import random
import itertools

sentences = []
sentences_tagged = []
validation_sentences = []
validation_sentences_tagged = []

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
# model.train(validation_sentences, total_examples=len(validation_sentences), epochs=model.epochs)

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
                    try:
                        similar_pos = list(neighbors_contextual_pos_for_word[word][previous_pos])
                        similar_pos.extend(similar_pos)
                        similar = model.wv.most_similar(word.lower())
                        for sim_word, sim in similar:
                            if sim_word in neighbors_contextual_pos_for_word:
                                try:
                                    options = neighbors_contextual_pos_for_word[sim_word][previous_pos]
                                except:
                                    options = list(
                                        itertools.chain.from_iterable(
                                            list(neighbors_contextual_pos_for_word[sim_word].values())))
                                finally:
                                    similar_pos.extend(max(Counter(options).items(), key=itemgetter(1))[0])
                        prediction = max(Counter(similar_pos).items(), key=itemgetter(1))[0]
                    except:
                        options = list(itertools.chain.from_iterable(list(neighbors_contextual_pos_for_word[word].values())))
                        prediction = max(Counter(options).items(), key=itemgetter(1))[0]
                    # finally:
                    #     # prediction = random.choice(options)
                    #     prediction = max(Counter(options).items(), key=itemgetter(1))[0]
                else:
                    # prediction = random.choices(missing_word_options, weights=weights, k=1)[0]
                    prediction = 'JJ'
                    missing += 1
                if prediction == actual:
                    success += 1
                previous_pos = prediction
                total_words += 1

print(f"success: {success}")
print(f"total_words: {total_words}")
print(f"success/total_words: {success/total_words}")
print(f"missing: {missing}")


# for sentence in sentences_tagged:
#     for word, tag in sentence:
#         most_similar = model.wv.most_similar(positive=word)


# from collections import defaultdict
# from operator import itemgetter
# import random
# import itertools
#
# pairs_count = defaultdict(lambda: 0)
# poss_count = defaultdict(lambda: 0)
# words_count = defaultdict(lambda: 0)
# words_pos_counts = defaultdict(lambda: defaultdict(lambda: 0))
# most_common_pos_for_word = {}
# neighbors_contextual_pos_for_word = defaultdict(lambda: defaultdict(list))
#
# with open("data/ass1-tagger-train") as train_file:
#     line = True
#     while line:
#         line = train_file.readline().split()
#         for index in range(len(line)):
#             pair = line[index].split('/')
#             left_pair = line[index - 1].split('/') if index > 0 else [None]
#             right_pair = line[index + 1].split('/') if index < len(line) - 1 else [None]
#             last, left_last, right_last = len(pair) - 1, len(left_pair) - 1, len(right_pair) - 1
#             pos, left_pos, right_pos = pair[last], left_pair[left_last], right_pair[right_last]
#             words = pair[:last]
#             for word in words:
#                 # word = word.lower()
#                 pairs_count[tuple([word, pos])] += 1
#                 poss_count[pos] += 1
#                 words_pos_counts[word][pos] += 1
#                 neighbors_contextual_pos_for_word[word][left_pos].append(pos)
#
#
# pairs_counts = dict(sorted(pairs_count.items(), key=itemgetter(1), reverse=False))
# poss_count = dict(poss_count)
# words_count = dict(words_count)
# words_pos_counts = {word: dict(words_pos_counts[word]) for word in words_pos_counts}
# neighbors_contextual_pos_for_word = {word: dict(neighbors_contextual_pos_for_word[word])
#                                      for word in neighbors_contextual_pos_for_word}
#
# for word, poss in words_pos_counts.items():
#     most_common_pos_for_word[word] = max(poss.items(), key=itemgetter(1))[0]
#
#
# total_words = 0
# success = 0
# missing = 0
#
# # with open("data/ass1-tagger-dev") as train_file:
# #     line = True
# #     while line:
# #         line = train_file.readline().split()
# #         for pair in line:
# #             split_pair = pair.split('/')
# #             last_index = len(split_pair) - 1
# #             actual = split_pair[last_index]
# #             for word in split_pair[:last_index]:
# #                 # word = word.lower()
# #                 total_words += 1
# #                 prediction = most_common_pos_for_word[word] if word in most_common_pos_for_word else 'JJ'
# #                 if word not in most_common_pos_for_word:
# #                     missing += 1
# #                 if prediction == actual:
# #                     success += 1
#
#
# with open("data/ass1-tagger-dev") as train_file:
#     line = True
#     missing_word_options = list(poss_count.keys())
#     total_appearances = sum(list(poss_count.values()))
#     weights = tuple([poss_count[pos]/total_appearances for pos in missing_word_options])
#     print(weights)
#     while line:
#         line = train_file.readline().split()
#         previous_pos = None
#         for index in range(len(line)):
#             pair = line[index].split('/')
#             last, left_last, right_last = len(pair) - 1, len(left_pair) - 1, len(right_pair) - 1
#             actual = pair[last]
#             for word in pair[:last]:
#                 # word = word.lower()
#                 total_words += 1
#                 if word in neighbors_contextual_pos_for_word:
#                     try:
#                         options = neighbors_contextual_pos_for_word[word][previous_pos]
#                     except:
#                         options = list(itertools.chain.from_iterable(
#                             list(neighbors_contextual_pos_for_word[word].values())))
#                     finally:
#                         prediction = random.choice(options)
#                 else:
#                     options = poss_count.keys()
#                     # prediction = random.choices(missing_word_options, weights=weights, k=1)[0]
#                     prediction = 'JJ'
#                     missing += 1
#                 if prediction == actual:
#                     success += 1
#                 previous_pos = prediction
#
# print(f"success: {success}")
# print(f"total_words: {total_words}")
# print(f"success/total_words: {success/total_words}")
# print(f"missing: {missing}")
