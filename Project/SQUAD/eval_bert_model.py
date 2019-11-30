import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from models.BertAnswerClassification import BertAnswerClassification
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
    "--models", action='append', default=None, help="Model"
)
parser.add_argument(
    "--file-path", default='data/dev-v2.0.json', help="Test/Dev data"
)
parser.add_argument(
    "--out-path", default='results/', help="Out path"
)
parser.add_argument(
    "--batch-size", type=int, default=3, help="Number of epochs"
)
parser.add_argument(
    "--model-paths", action='append', default=None, help="Path to saved model state"
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    models = []

    for model_name in args.models:
        if model_name == 'BertAnswerClassification': # Classification case
            model = BertAnswerClassification(distil = False) # Distil version not available yet
            models.append(model)
        elif model_name == 'BertQABase': # Question answering case
            model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
            models.append(model)
        elif model_name == 'DistilBertQABase':
            model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
            models.append(model)
        else:
            assert False, 'Unknown model ' + model_name

    for model_index in range(0, len(args.models)):
        models[model_index].load_state_dict(torch.load(args.model_paths[model_index]))

    for model in models:
        model.eval()
    
    print('Loading dataset')
    dataset, examples, features = load_and_cache_examples(args.file_path, distil, tokenizer, evaluate=True)
    print('Finished loading dataset')

    sampler = SequentialSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    results = []
    iterator = tqdm(iter(train_dataloader))
    for batch_num, batch in tqdm(enumerate(iterator)):
        with torch.no_grad():
            outputs = []
            for model in models:
                outputs.append(model(batch[0], attention_mask=batch[1]))

        example_indices = batch[3]
        for i, example_index in enumerate(example_indices):
            unique_index = int(features[example_index.item()].unique_id)

            start_logits = torch.zeros_like(outputs[0][0][i])
            end_logits = torch.zeros_like(outputs[0][1][i])

            for output in outputs:
                start_logits += output[0][i]
                end_logits += output[1][i]
            
            result_here = RawResult(unique_id=unique_index, start_logits=start_logits, end_logits=end_logits)
            results.append(result_here)
    
    prediction_file = os.path.join(args.out_path, "predictions.json")
    nbest_file = os.path.join(args.out_path, "nbest_predictions.json")
    null_log_odds_file = os.path.join(args.out_path, "null_odds.json")
    
    write_predictions(examples, features, results, n_best_size=20,
                    max_answer_length=30, do_lower_case=True, output_prediction_file=prediction_file, 
                    output_nbest_file=nbest_file, output_null_log_odds_file=null_log_odds_file, verbose_logging=False,
                    version_2_with_negative=True, null_score_diff_threshold=0)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)



