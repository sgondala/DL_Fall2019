import torch
import torch.nn as nn
import pickle

class SimpleFusionModel(nn.Module):
    """
    Represent both question and text in vector form and score = linear_layer(q_question' * q_text)
    """
    def __init__(self, embed_size, hidden_size, out_size = 100, num_layers = 1, pretrained_embeddings = None, vocab_size = 400000):
        super(SimpleFusionModel, self).__init__()
        self.model_embeddings = None
        if pretrained_embeddings is not None:
            self.model_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.model_embeddings = nn.Embedding(vocab_size, embed_size)
        self.questions_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.text_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.linear_layer = nn.Linear(4*num_layers*hidden_size, out_size)

    def forward(self, questions, texts, sigmoid=False):
        """
        questions, texts - b * length * embed_size
        """
        batch_size = questions.shape[0]
        _, (h_questions, _) = self.questions_lstm(self.model_embeddings(questions)) # num_layers * num_directions, batch_size, hidden_size
        _, (h_text, _) = self.text_lstm(self.model_embeddings(texts)) # num_layers * num_directions, batch_size, hidden_size
        h_questions = h_questions.transpose(0,1).reshape((batch_size, -1)) # batch_size, num_layers * num_directions * hidden_size
        h_text = h_text.transpose(0,1).reshape((batch_size, -1)) # batch_size, num_layers * num_directions * hidden_size
        concat = torch.cat((h_questions, h_text), 1) # batch_size, 2 * num_layers * num_directions * hidden_size
        output = self.linear_layer(concat) # batch_size, out_size
        return output