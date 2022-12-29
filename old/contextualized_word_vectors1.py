import gensim
import gensim.downloader as dl
from collections import defaultdict, Counter
from operator import itemgetter
import numpy as np
from numpy.linalg import norm
import itertools


import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


""" 1.1 - Embedding vectors for "am" and "<mask>" """
def get_word_idx(sent: str, word: str):
    # print(sent)
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
       Select only those subword token outputs that belong to our word of interest
       and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def get_word_vector(sent, idx, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

    return get_hidden_states(encoded, token_ids_word, model, layers)


def get_word_embedding(sentence, word, layers=[-4, -3, -2, -1]):
    # Use last four layers by default
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    idx = get_word_idx(sentence, word)

    word_embedding = get_word_vector(sentence, idx, tokenizer, model, layers)

    return word_embedding


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
sentences, sentences_tagged = get_sen_and_tagged_sen("../data/ass1-tagger-train")
poss_count = defaultdict(lambda: 0)
neighbors_contextual_pos_for_word = defaultdict(lambda: defaultdict(list))
words_poss = defaultdict(set)
# model = dl.load("word2vec-google-news-300")
model = dl.load("glove-twitter-200")
prev_pos_next_pos_counts = defaultdict(lambda: defaultdict(lambda: 0))
pos_avg_vectors = dict()
pos_count = dict()
words_poss = defaultdict(set)
word_embeddings = dict()

for sentence in sentences_tagged:
    for index in range(len(sentence)):
        word, pos = sentence[index]
        previous_word, previous_pos = sentence[index - 1] if index > 0 else (None, None)
        poss_count[pos] += 1
        neighbors_contextual_pos_for_word[word][previous_pos].append(pos)
        prev_pos_next_pos_counts[previous_pos][pos] += 1

for sentence in sentences_tagged:
    sent_str = ""
    for word, pos in sentence:
        sent_str += f"{word} "
    sent_str = sent_str[:len(sent_str)-1]
    for word, pos in sentence:
        if pos in pos_avg_vectors:
            pos_avg_vectors[pos] += get_word_embedding(sent_str, word)
            pos_count[pos] += 1
        else:
            pos_avg_vectors[pos] = get_word_embedding(sent_str, word)
            pos_count[pos] = 1

for pos in pos_avg_vectors:
    pos_avg_vectors[pos] /= pos_count[pos]

poss_count = dict(poss_count)
neighbors_contextual_pos_for_word = {word: dict(neighbors_contextual_pos_for_word[word])
                                     for word in neighbors_contextual_pos_for_word}

missing_word_options = list(poss_count.keys())
total_appearances = sum(list(poss_count.values()))
weights = tuple([poss_count[pos]/total_appearances for pos in missing_word_options])

""" Validation """
total_words = 0
success = 0
missing = 0
validation_sentences, validation_sentences_tagged = get_sen_and_tagged_sen("../data/ass1-tagger-dev")
# model = gensim.models.Word2Vec(sentences + validation_sentences, min_count=1)


def get_best_prediction_no_vectors(word, previous_pos):
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


def get_best_prediction(word, previous_pos, sentence):
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
        try:
            word_vec = get_word_embedding(sentence, word)
            pos_vectors_sim = []
            for pos in pos_avg_vectors:
                pos_vec = pos_avg_vectors[pos]
                cosine = np.dot(word_vec, pos_vec) / (norm(word_vec) * norm(pos_vec))
                pos_vectors_sim.append((pos, cosine))
            closest_pos_vector = max(pos_vectors_sim, key=itemgetter(1))[0]
            prediction = closest_pos_vector
        except:
            prediction = max(prev_pos_next_pos_counts[previous_pos].items(), key=itemgetter(1))[0]

    return prediction


for sentence in validation_sentences_tagged:
    sent_str = ""
    for word, pos in sentence:
        sent_str += f"{word} "
    sent_str = sent_str[:len(sent_str)-1]
    previous_pos = None
    for index in range(len(sentence)):
        word, actual = sentence[index]
        prediction = get_best_prediction(word, previous_pos, sent_str)
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
