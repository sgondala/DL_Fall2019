import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from scripts.utils import *
from models.SimpleFusionModel import SimpleFusionModel
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
    "--path", default='data/dev-v2.0.json', help="Train data"
)
parser.add_argument(
    "--vectors-path", default='data/glove_vectors.pkl', help="Embedding data"
)
parser.add_argument(
    "--indices-path", default='data/glove_token_to_index.pkl', help="Indices data"
)
parser.add_argument(
    "--start-path", default='start.pth', help="Start path"
)
parser.add_argument(
    "--end-path", default='end.pth', help="End path"
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
    data = flatten_squad_data(args.path, train_data=False)
    print("Data length", len(data))
    embedding_vector_file = open(args.vectors_path,"rb")
    embedding_vectors = pickle.load(embedding_vector_file)
    tokens_to_id_file = open(args.indices_path, "rb")
    tokens_to_index = pickle.load(tokens_to_id_file)
    print("Data loading finished...")

    start_model = SimpleFusionModel(300, args.hidden_layer, pretrained_embeddings = embedding_vectors).to(device)
    end_model = SimpleFusionModel(300, args.hidden_layer, pretrained_embeddings = embedding_vectors).to(device)
    
    start_model.load_state_dict(torch.load(args.start_path))
    end_model.load_state_dict(torch.load(args.end_path))

    print("Loaded saved model parameters")
    start_model.eval()
    end_model.eval()

    final_entries = {}
    batch_size = args.batch_size
    number_of_iterations = len(data) // batch_size 
    number_of_iterations += 1 if len(data) % batch_size != 0 else 0

    start_indices_all = torch.Tensor([]).long().to(device)
    end_indices_all = torch.Tensor([]).long().to(device)
    example_ids_all = []

    for iteration_number in tqdm(range(number_of_iterations)):
        questions, contexts, _, _, example_ids = get_mini_batch(data, tokens_to_index, 
            start_index=iteration_number*batch_size, 
            end_index=(iteration_number+1)*batch_size, device=device, batch_size=args.batch_size, train_data=False)

        start_indices = start_model(questions, contexts).max(1).indices
        end_indices = end_model(questions, contexts).max(1).indices
        start_indices_all = torch.cat((start_indices_all, start_indices), 0)
        end_indices_all = torch.cat((end_indices_all, end_indices), 0)
        example_ids_all += example_ids

    print(example_ids_all)
    results = construct_results(data, start_indices_all, end_indices_all, example_ids_all)
    with open('results.json', 'w') as outfile:
        json.dump(results, outfile)