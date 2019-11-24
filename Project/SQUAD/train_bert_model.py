import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scripts.utils import *
from models.SimpleFusionModel import SimpleFusionModel
import argparse
from torch.utils.tensorboard import SummaryWriter
from scripts.utils_squad_evaluate import *
from scripts.utils_squad import *
from transformers import *
from torch.utils.data import *

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser("SQUAD parser")
parser.add_argument(
    "--model", default='BertQABase', help="Model"
)
parser.add_argument(
    "--num-epochs", type=int, default=2, help="Number of epochs"
)
parser.add_argument(
    "--lr", type=float, default=3e-5, help="Learning rate"
)
parser.add_argument(
    "--train-path", default='data/train-v2.0.json', help="Train data"
)
parser.add_argument(
    "--out-path", default='checkpoints/', help="Out path"
)
parser.add_argument(
    "--batch-size", type=int, default=3, help="Number of epochs"
)
parser.add_argument(
    "--train-percentage", type=int, default=10, help="Percentage of data to train on"
)

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Loading dataset')
    dataset = load_and_cache_examples(args.train_path, tokenizer)
    print('Finished loading dataset')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Training only on 10% of data
    indices = list(np.random.randint(len(dataset), size=len(dataset) // args.train_percentage))
    sampler = SubsetRandomSampler(indices)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    
    for _ in range(args.num_epochs):
        iterator = tqdm(iter(train_dataloader))
        for batch_num, batch in tqdm(enumerate(train_dataloader)):
            output = model(batch[0], attention_mask=batch[1], start_positions=batch[3], end_positions=batch[4])
            loss = output[0]
            writer.add_scalar('Bert base loss', loss, batch_num)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
    
    writer.close()
    torch.save(model.state_dict(), args.out_path + args.model + '.pth')

    

