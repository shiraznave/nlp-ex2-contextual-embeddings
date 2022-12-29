from collections import Counter, defaultdict
from operator import itemgetter
import torch
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)


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


def get_end_idx(word, start_idx):
    word_tokens = tokenizer.tokenize(f" {word}")
    end_idx = start_idx + len(word_tokens)
    return end_idx


def get_word_embedding(sent, start_idx, end_idx):
    encoded = tokenizer(sent, return_tensors="pt")
    output = model(**encoded)
    word_vector = output.last_hidden_state[0][start_idx]
    for index in range(start_idx + 1, end_idx):
        word_vector += output.last_hidden_state[0][index]
    word_vector /= end_idx - start_idx
    return word_vector.detach().numpy()


def get_similarity(tensor1, tensor2):
    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(tensor1, tensor2)
    return similarity


""" Train """
train_sentences, train_sentences_tagged = get_sen_and_tagged_sen("data/ass1-tagger-train")
word_pos_avg_vectors = defaultdict(dict)
pos_count = dict()
pos_avg_vectors = dict()


def create_avg_vectors():
    """ For missing words """
    global pos_avg_vectors
    iteration = 0
    for sentence in train_sentences_tagged:
        sent_str = " "
        for index in range(len(sentence)):
            word, pos = sentence[index]
            sent_str += f"{word} "
        sent_str = sent_str[:len(sent_str)-1]
        start_idx = 1
        for index in range(len(sentence)):
            word, pos = sentence[index]
            end_idx = get_end_idx(word, start_idx)
            word_vec = get_word_embedding(sent_str, start_idx, end_idx)
            if pos in pos_avg_vectors:
                pos_avg_vectors[pos] += word_vec
                pos_count[pos] += 1
            else:
                pos_avg_vectors[pos] = word_vec
                pos_count[pos] = 1

            if pos in word_pos_avg_vectors[word]:
                cur_vec, cur_count = word_pos_avg_vectors[word][pos]
                word_pos_avg_vectors[word][pos] = (cur_vec + word_vec, cur_count + 1)
            else:
                word_pos_avg_vectors[word][pos] = (word_vec, 1)


            start_idx = end_idx

        iteration += 1
        # print(sent_str)
        print(f"sentence {iteration} out of {len(train_sentences_tagged)}")

    for pos in pos_avg_vectors:
        pos_avg_vectors[pos] /= pos_count[pos]


def create_word_pos_avg_vectors():
    """ With Contextualized Vectors """
    global word_pos_avg_vectors
    iteration = 0
    for sentence in train_sentences_tagged:
        sent_str = " "
        for word, pos in sentence:
            sent_str += f"{word} "
        sent_str = sent_str[:len(sent_str)-1]
        start_idx = 1
        for index in range(len(sentence)):
            word, pos = sentence[index]
            end_idx = get_end_idx(word, start_idx)
            word_vec = get_word_embedding(sent_str, start_idx, end_idx)
            if pos in word_pos_avg_vectors[word]:
                cur_vec, cur_count = word_pos_avg_vectors[word][pos]
                word_pos_avg_vectors[word][pos] = (cur_vec + word_vec, cur_count + 1)
            else:
                word_pos_avg_vectors[word][pos] = (word_vec, 1)
            start_idx = end_idx

        iteration += 1
        print(f"sentence {iteration} out of {len(train_sentences_tagged)}")

    word_pos_avg_vectors = dict(word_pos_avg_vectors)
    for word in word_pos_avg_vectors:
        word_pos_avg_vectors[word] = dict(word_pos_avg_vectors[word])
        for pos in word_pos_avg_vectors[word]:
            sum_vector, count = word_pos_avg_vectors[word][pos]
            word_pos_avg_vectors[word][pos] = sum_vector/count


""" Validation """
total_words = 0
success = 0
missing = 0
validation_sentences, validation_sentences_tagged = get_sen_and_tagged_sen("data/ass1-tagger-dev")


def get_best_prediction(sentence, word, start_idx, end_idx):
    word_vec = get_word_embedding(sentence, start_idx, end_idx)
    pos_vectors_sim = []
    if word in word_pos_avg_vectors:
        for pos, rel_pos_vec in word_pos_avg_vectors[word].items():
            sim = get_similarity(word_vec, rel_pos_vec)
            pos_vectors_sim.append((pos, sim))
    if len(pos_vectors_sim) == 0:
        for pos, pos_vec in pos_avg_vectors.items():
            pos_vectors_sim.append((pos, sim))
    return max(pos_vectors_sim, key=itemgetter(1))[0]


def run_validation():
    global total_words, success
    for sentence in validation_sentences_tagged:
        sent_str = " "
        for word, pos in sentence:
            sent_str += f"{word} "
        sent_str = sent_str[:len(sent_str)-1]
        previous_pos = None
        start_idx = 1
        for index in range(len(sentence)):
            word, actual = sentence[index]
            end_idx = get_end_idx(word, start_idx)
            prediction = get_best_prediction(word, index, sent_str)
            if prediction == actual:
                success += 1
            else:
                if word in word_pos_avg_vectors:
                    print("seen", previous_pos, word, prediction, actual,
                          {pos: Counter(options) for pos, options in word_pos_avg_vectors[word].items()})
                else:
                    print("unseen", previous_pos, word, prediction, actual)
            previous_pos = prediction
            start_idx = end_idx
            total_words += 1


import pickle
create_avg_vectors()
print("done create_pos_avg_vectors")
pickle.dump(pos_avg_vectors, open("pos_avg_vectors.pkl", "wb"))
# create_word_pos_avg_vectors()
print("done create_word_pos_avg_vectors")
pickle.dump(word_pos_avg_vectors, open("word_pos_avg_vectors.pkl", "wb"))
run_validation()


print(f"success: {success}")
print(f"total_words: {total_words}")
print(f"success/total_words: {success/total_words}")
print(f"missing: {missing}")
