import torch
import torch.nn as nn
import torch.optim as optim
from transformers import *


class BertAnswerClassification(nn.Module):
    def __init__(self, distil = False):
        super(BertAnswerClassification, self).__init__()
        if distil == True:
            assert False, 'Distil version not available in answer classification'
            self.bert_model = DistilBertForNextSentencePrediction.from_pretrained('distilbert-base-uncased')
        else:
            self.bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, start_positions, end_positions):
        """
        0 if answer is present in the text
        1 otherwise 
        """
        bool_tensor = (start_positions == 0) & (end_positions == 0)
        labels_tensor = bool_tensor.type(torch.long)
        return self.bert_model(input_ids, attention_mask = attention_mask, next_sentence_label = labels_tensor)

