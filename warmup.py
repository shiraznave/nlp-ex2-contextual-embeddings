import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel


""" 1.1 - Embedding vectors for "am" and "<mask>" """
def get_word_idx(tokenizer, sent: str, word: str):
    tokens = tokenizer.tokenize(sent)
    print(tokens)
    word_tokens = tokenizer.tokenize(f" {word}")[0]
    return tokens.index(word_tokens) + 1


def get_word_vector(sent, start_idx, end_idx, tokenizer, model):
    encoded = tokenizer(sent, return_tensors="pt")
    output = model(**encoded)
    word_vector = output.last_hidden_state[0][start_idx]
    print(start_idx)
    for index in range(start_idx + 1, end_idx):
        word_vector += output.last_hidden_state[0][index]
        print(index)
    word_vector /= end_idx - start_idx
    return word_vector


def get_word_embedding(sentence, word):
    sentence = f" {sentence}"
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)
    start_idx = get_word_idx(tokenizer, sentence, word)
    # input = word if start_idx == 1 else f" {word}"
    word_tokens = tokenizer.tokenize(f" {word}")
    end_idx = start_idx + len(word_tokens)

    word_embedding = get_word_vector(sentence, start_idx, end_idx, tokenizer, model)

    return word_embedding

import numpy as np
sentence = "I am so <mask>"
print((get_word_embedding(sentence, "am").detach().numpy()))
# print(get_word_embedding(sentence, "<mask>"))


# """ 1.2 - Top 5 word prediction for "am" and for "<mask>" """
# from transformers import pipeline
# unmasker = pipeline('fill-mask', model='roberta-base')
# print(unmasker("I am so <mask>"))
# print(unmasker("I <mask> so <mask>"))


# """ 2 - Tow sentence with a common word, high similarity """
# sentence1 = "I love you"
# tensor1 = get_word_embedding(sentence1, "love")
# sentence2 = "I love him"
# tensor2 = get_word_embedding(sentence2, "love")
# cos = torch.nn.CosineSimilarity(dim=0)
# similarity = cos(tensor1, tensor2)
# print(similarity)


# """ 3 - Tow sentence with a common word, low similarity """
# sentence1 = "I'll have to go back - I think I've left the iron turned on"
# tensor1 = get_word_embedding(sentence1, "left")
# sentence2 = "Hold your fork in your left hand and your knife in your right hand"
# tensor2 = get_word_embedding(sentence2, "left")
# cos = torch.nn.CosineSimilarity(dim=0)
# similarity = cos(tensor1, tensor2)
# print(similarity)

""" 4 - sentence with n words, m>n tokens """
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokens = tokenizer.tokenize("I'm going to feed my sister’s cat")
print(tokens)



# tokens = tokenizer.tokenize(" Goodman")
# print(tokens)
# sentence = "In an Oct. 19 review of `` The Misanthrope '' at Chicago 's Goodman Theatre ( `` Revitalized Classics Take the Stage in Windy City , '' Leisure & Arts ) , the role of Celimene , played by Kim Cattrall , was mistakenly attributed to Christina Haag ."
# tokens = tokenizer.tokenize("In an Oct. 19 review of `` The Misanthrope '' at Chicago 's Goodman Theatre ( `` Revitalized Classics Take the Stage in Windy City , '' Leisure & Arts ) , the role of Celimene , played by Kim Cattrall , was mistakenly attributed to Christina Haag .")
# full_sen_tokens_len = len(tokens)
# print(tokens)
# print(tokenizer.tokenize("Shiraz is dancing"))
# print(tokenizer.tokenize("Shiraz"))
# sentence = "Shiraz is dancing"
# print(len(get_word_embedding(sentence, "Shiraz")))
#
#
#
