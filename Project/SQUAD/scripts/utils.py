import json
import numpy as np
from nltk import word_tokenize
import torch
import string

def flatten_squad_data(json_file_name, train_data=True):
    f = open(json_file_name)
    json_data = json.load(f)['data']
    retval = []
    field_list = ['title', 'context', 'question', 'id_here', 'is_impossible', 'text', 'start_token', 'end_token']
    test_fields = ['context', 'question', 'id_here']
    for data_point in json_data:
        title = data_point['title'].lower()
        for paragraph in data_point['paragraphs']:
            context = paragraph['context'].lower()
            for qas in paragraph['qas']:
                question = qas['question'].lower()
                id_here = qas['id'].lower()
                is_impossible = qas['is_impossible']
                if train_data == False:
                    retval.append(dict(zip(test_fields, [context, question, id_here])))
                else:
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
    max_length = 100, device=torch.device('cpu'),
    start_index = None, end_index = None, train_data = True):

    """
    Data - list of python dictionaries
    Padding questions and texts to have size 100
    Returns torch tensor with indices
    """
    max_index = len(data)
    if start_index == None and end_index == None:
        indices = np.random.randint(0, max_index, size=batch_size)
        indices = [indices[index] for index in range(len(indices)) if data[indices[index]]['end_token'] < max_length]
    else:
        indices = range(min(len(data), start_index), min(len(data), end_index))

    questions = [sentence_to_indices(tokenize_and_pad_sentence(data[i]['question'], pad_token, 100), tokens_to_index) for i in indices]
    contexts = [sentence_to_indices(tokenize_and_pad_sentence(data[i]['context'], pad_token, 100), tokens_to_index) for i in indices]

    questions = torch.Tensor(questions).long().to(device)
    contexts = torch.Tensor(contexts).long().to(device)
    example_ids = [data[index]['id_here'] for index in indices]
    
    if train_data == True:
        start_tokens = torch.Tensor([data[index]['start_token'] for index in indices]).long().to(device)
        end_tokens = torch.Tensor([data[index]['end_token'] for index in indices]).long().to(device)
    else:
        start_tokens = []
        end_tokens = []
    
    return questions, contexts, start_tokens, end_tokens, example_ids

def construct_results(data, start_indices, end_indices, example_ids):
    retval = {}
    for i in range(len(example_ids)):
        start_index = int(start_indices[i].item())
        end_index = int(end_indices[i].item())
        retval[example_ids[i]] = ' '.join(data[i]['context'].split(' ')[start_index:end_index])
    return retval

if __name__ == "__main__":
    data = flatten_squad_data('data/train-v2.0.json')
    mini_batch = get_mini_batch(data)
    assert mini_batch.shape[0] == 32
