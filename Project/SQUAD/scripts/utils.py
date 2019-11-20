import json
import numpy as np
from nltk import word_tokenize
import torch
import string

def flatten_squad_data(json_file_name):
    f = open(json_file_name)
    json_data = json.load(f)['data']
    retval = []
    field_list = ['title', 'context', 'question', 'id_here', 'is_impossible', 'text', 'start_token', 'end_token']
    for data_point in json_data:
        title = data_point['title'].lower()
        for paragraph in data_point['paragraphs']:
            context = paragraph['context'].lower()
            for qas in paragraph['qas']:
                question = qas['question'].lower()
                id_here = qas['id'].lower()
                is_impossible = qas['is_impossible']
                for answer in qas['answers']:
                    text = answer['text'].lower()
                    answer_start = answer['answer_start']
                    start_token = len(context[:answer_start].split())
                    end_token = len(context[:answer_start + len(text)].split())
                    retval.append(dict(zip(field_list, [title, context, question, id_here, is_impossible, text, start_token, end_token])))
    return retval

def tokenize_and_pad_sentence(sentence, pad_token, final_length):
    # sentence = sentence.split()
    sentence = sentence.split()[:final_length]
    while (len(sentence) < final_length):
        sentence.append(pad_token)
    return sentence

def sentence_to_indices(sentence, tokens_to_index, unknown_token='@@UNK@@'):
    index_of_unk = tokens_to_index[unknown_token]
    return [tokens_to_index.get(word.strip(string.punctuation), index_of_unk) for word in sentence]

def get_mini_batch(data, tokens_to_index, batch_size=32, 
    pad_token='@@PAD@@', unknown_token='@@UNK@@', 
    max_length = 100, device=torch.device('cpu')):

    """
    Data - list of python dictionaries
    Padding questions and texts to have size 100
    Returns torch tensor with indices
    """
    max_index = len(data)
    indices = np.random.randint(0, max_index, size=batch_size)
    # indices[index] for index in range(len(indices)) if token[index] < 8
    # print(indices)
    # print([data[index]['end_token'] for index in range(len(indices))])
    indices = [indices[index] for index in range(len(indices)) if data[indices[index]]['end_token'] < max_length]
    # print([data[index]['end_token'] for index in range(len(indices))])
    # assert False
    # print(indices)

    questions = [sentence_to_indices(tokenize_and_pad_sentence(data[i]['question'], pad_token, 100), tokens_to_index) for i in indices]
    contexts = [sentence_to_indices(tokenize_and_pad_sentence(data[i]['context'], pad_token, 100), tokens_to_index) for i in indices]

    questions = torch.Tensor(questions).long().to(device)
    contexts = torch.Tensor(contexts).long().to(device)
    start_tokens = torch.Tensor([data[index]['start_token'] for index in indices]).long().to(device)
    end_tokens = torch.Tensor([data[index]['end_token'] for index in indices]).long().to(device)
    example_ids = [data[index]['id_here'] for index in indices]

    return questions, contexts, start_tokens, end_tokens, example_ids


# def get_contiguous_mini_batch_for_test(data, tokens_to_index, start_index, end_index, batch_size=32, pad=True, pad_token='@@PAD@@', unknown_token='@@UNK@@', device=torch.device('cpu')):
#     """
#     Padding questions and texts to have size 100
#     Returns torch tensor with indices

#     This should return a question, along with all the long answer candidates
#     """
#     max_index = data.shape[0]
#     indices = range(start_index, end_index)
#     # indices = [index in indices where data[index]['end_token'] <= ]
#     questions = []
#     texts = []
#     if pad == True:
#         questions = [sentence_to_indices(pad_sentence(data[i]['question'], pad_token, 100), tokens_to_index) for i in indices]
#         texts = [sentence_to_indices(pad_sentence(data[i]['text'], pad_token, 100), tokens_to_index) for i in indices]
#     else:
#         questions = [sentence_to_indices(data[i]['question'], tokens_to_index) for i in indices]
#         texts = [sentence_to_indices(data[i]['text'], tokens_to_index) for i in indices]
#     questions = torch.Tensor(questions).long().to(device)
#     texts = torch.Tensor(texts).long().to(device)
#     labels = torch.Tensor([data[i]['label'] for i in indices]).unsqueeze(1).float().to(device)
#     return questions, texts, labels

if __name__ == "__main__":

    data = flatten_squad_data('data/train-v2.0.json')
    mini_batch = get_mini_batch(data)
    assert mini_batch.shape[0] == 32
