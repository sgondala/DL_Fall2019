import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scripts.utils import *
from models.simple_attention_model import SimpleAttentionModel
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser("NQA train parser")
parser.add_argument(
    "--model", default='SAM', help="Model"
)
parser.add_argument(
    "--num-epochs", type=int, default=10, help="Number of epochs"
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate"
)
parser.add_argument(
    "--hidden-layer", type=int, default=100, help="Hidden layer dimension"
)
parser.add_argument(
    "--path", default='datasets/train_10k.json', help="Train data"
)
parser.add_argument(
    "--vectors-path", default='datasets/glove_vectors.pkl', help="Embedding data"
)
parser.add_argument(
    "--indices-path", default='datasets/glove_token_to_index.pkl', help="Indices data"
)
parser.add_argument(
    "--in-path", default='checkpoints/final_model.pth', help="Out path"
)
parser.add_argument(
    "--batch-size", type=int, default=32, help="Number of epochs"
)

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    print("Loading data...")
    data = complete_in_memory_data_loader(args.path) # 
    embedding_vector_file = open(args.vectors_path,"rb")
    embedding_vectors = pickle.load(embedding_vector_file)
    tokens_to_id_file = open(args.indices_path, "rb")
    tokens_to_index = pickle.load(tokens_to_id_file)
    print("Data loading finished...")

    # number_of_iterations = len(data) // args.batch_size
    model = SimpleAttentionModel(300, args.hidden_layer, pretrained_embeddings=embedding_vectors).to(device)
    model.load_state_dict(torch.load(args.in_path))
    print("Loaded saved model parameters")
    model.eval()

    number_of_iterations = len(data)
    final_entries = []

    # i = 0
    for iteration_number in tqdm(range(number_of_iterations)):
        # i += 1
        # if i == 10:
        #     break
        data_point = data[iteration_number]
        batch_size = len(data_point['labels'])

        question = sentence_to_indices(pad_sentence(data_point['question'], '@@PAD@@', 100), tokens_to_index)
        questions = torch.Tensor([question]*batch_size).long().to(device)
        long_answer_candidates = \
            [sentence_to_indices(pad_sentence(data_point['long_answer_candidates'][i], '@@PAD@@', 100), tokens_to_index) for i in range(len(data_point['long_answer_candidates']))]
        long_answer_candidates = torch.Tensor(long_answer_candidates).long().to(device)

        out = model(questions, long_answer_candidates, sigmoid=True).squeeze()
        labels = np.array(data_point['labels'])
        correct_answer = -1 if labels.sum() == 0 else np.argmax(labels)
        if correct_answer == -1:
            continue
        final_entries.append({'answer_index':correct_answer, 'scores':out.data.cpu().numpy().tolist()})
    
    pickle.dump(final_entries, open('final_entries.p', 'wb'))