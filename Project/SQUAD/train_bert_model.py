import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from models.BertAnswerClassification import BertAnswerClassification
from torch.utils.tensorboard import SummaryWriter
from scripts.utils_squad_evaluate import *
from scripts.utils_squad import *
from transformers import *
from torch.utils.data import *
from scripts.utils import *

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser("SQUAD parser")
parser.add_argument(
    "--model", default=None, help="Model"
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
parser.add_argument(
    "--distil", type=bool, default=False, help="Use distil bert"
)
parser.add_argument(
    "--saved-model", default=None, help="Path to saved model"
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed"
)

args = parser.parse_args()
# Reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()

def save_checkpoint(args, checkpoint_number):
    output_file_name = args.out_path + args.model + "_" + str(checkpoint_number) + "_" + str(args.train_percentage)
    if args.distil == True:
        output_file_name += "_distil"
    torch.save(model.state_dict(), output_file_name + '.pth')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    if args.distil == True:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if args.model == 'BertAnswerClassification': # Classification case
        model = BertAnswerClassification(distil = False) # Distil version not available yet
    elif args.model == 'BertQABase': # Question answering case
        if args.distil == True:
            model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
        else:
            model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    else:
        assert False, 'Unknown model'
    
    if args.saved_model is not None:
        print("Loading presaved weights")
        model.load_state_dict(torch.load(args.saved_model))
    else:
        print("Training from scratch")

    model.train()
    print('Loading dataset')
    dataset = load_and_cache_examples(args.train_path, args.distil, tokenizer)
    print('Finished loading dataset')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # By default only on 10% of data
    indices = list(np.random.choice(len(dataset), size=(len(dataset) * args.train_percentage) // 100, replace=False))
    sampler = SubsetRandomSampler(indices)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    if args.model == 'BertQABase':
        loss_scalar = 'Bert base loss' if args.distil is False else 'Distil bert base loss'
    elif args.model == 'BertAnswerClassification':
        loss_scalar = 'Bert classification loss' if args.distil is False else 'Distil bert classification loss'

    for epoch_number in range(args.num_epochs):
        save_checkpoint(args, epoch_number)
        iterator = tqdm(iter(train_dataloader))
        for batch_num, batch in tqdm(enumerate(iterator)):
            output = model(batch[0], attention_mask=batch[1], start_positions=batch[3], end_positions=batch[4])
            loss = output[0]
            writer.add_scalar(loss_scalar, loss, batch_num)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

    save_checkpoint(args, args.num_epochs)
    writer.close()
    
