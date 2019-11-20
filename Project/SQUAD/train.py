import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scripts.utils import *
from models.SimpleFusionModel import SimpleFusionModel
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser("SQUAD parser")
parser.add_argument(
    "--model", default='SFM', help="Model"
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
    "--path", default='data/train-v2.0.json', help="Train data"
)
parser.add_argument(
    "--vectors-path", default='data/glove_vectors.pkl', help="Embedding data"
)
parser.add_argument(
    "--indices-path", default='data/glove_token_to_index.pkl', help="Indices data"
)
parser.add_argument(
    "--out-path", default='checkpoints/', help="Out path"
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
    data = flatten_squad_data(args.path)
    embedding_vector_file = open(args.vectors_path,"rb")
    embedding_vectors = pickle.load(embedding_vector_file)
    tokens_to_id_file = open(args.indices_path, "rb")
    tokens_to_index = pickle.load(tokens_to_id_file)
    print("Data loading finished...")

    number_of_iterations = len(data) // 32

    start_model = SimpleFusionModel(300, args.hidden_layer, pretrained_embeddings = embedding_vectors).to(device)
    start_criterion = nn.CrossEntropyLoss(reduction='sum')
    start_optimizer = optim.Adam(start_model.parameters(), lr=args.lr)
    
    number_of_iterations = args.num_epochs * number_of_iterations

    for iteration_number in tqdm(range(number_of_iterations)):
        questions, contexts, start_tokens, _, _ = get_mini_batch(data, tokens_to_index, device=device, batch_size=args.batch_size)
        # print(start_tokens)
        start_out = start_model(questions, contexts)
        start_loss = start_criterion(start_out, start_tokens)
        if iteration_number % 100 == 0:
            print("Start loss", start_loss.data)
        start_loss.backward()
        nn.utils.clip_grad_norm_(start_model.parameters(), 2.0)
        start_optimizer.step()
        start_optimizer.zero_grad()
        start_model.zero_grad()

    end_model = SimpleFusionModel(300, args.hidden_layer, pretrained_embeddings = embedding_vectors).to(device)
    end_criterion = nn.CrossEntropyLoss(reduction='sum')
    end_optimizer = optim.Adam(end_model.parameters(), lr=args.lr)

    for iteration_number in tqdm(range(number_of_iterations)):
        questions, contexts, _, end_tokens, _ = get_mini_batch(data, tokens_to_index, device=device, batch_size=args.batch_size)
        end_out = end_model(questions, contexts)
        end_loss = end_criterion(end_out, end_tokens)
        if iteration_number % 100 == 0:
            print("Start loss", end_loss.data)
        end_loss.backward()
        nn.utils.clip_grad_norm_(end_model.parameters(), 2.0)
        end_optimizer.step()
        end_optimizer.zero_grad()
        end_model.zero_grad()
        
    torch.save(start_model.state_dict(), args.out_path + 'start.pth')
    torch.save(end_model.state_dict(), args.out_path + 'end.pth')