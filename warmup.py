import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


""" 1.1 - Embedding vectors for "am" and "<mask>" """
def get_word_idx(sent: str, word: str):
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


# sentence = "I am so <mask>"
# print(get_word_embedding(sentence, "am"))
# print(get_word_embedding(sentence, "<mask>"))


""" 1.2 - Top 5 word prediction for "am" and for "<mask>" """
# from transformers import pipeline
# unmasker = pipeline('fill-mask', model='roberta-base')
# print(unmasker("I am so <mask>"))
# print(unmasker("I <mask> so <mask>"))


""" 2 - Tow sentence with a common word, high similarity """
# sentence1 = "I love you"
# tensor1 = get_word_embedding(sentence1, "love")
# sentence2 = "I love him"
# tensor2 = get_word_embedding(sentence2, "love")
# cos = torch.nn.CosineSimilarity(dim=0)
# similarity = cos(tensor1, tensor2)
# print(similarity)


""" 3 - Tow sentence with a common word, low similarity """
# sentence1 = "It's very kind of you to invite me to your birthday party."
# tensor1 = get_word_embedding(sentence1, "kind")
# sentence2 = "What kind of food should I be feeding my dog?"
# tensor2 = get_word_embedding(sentence2, "kind")
# cos = torch.nn.CosineSimilarity(dim=0)
# similarity = cos(tensor1, tensor2)
# print(similarity)

""" 4 - sentence with n words, m>n tokens """
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokens = tokenizer.tokenize("I'm going to feed my sister's cat")
print(tokens)

