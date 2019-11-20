# Adapted from https://github.com/nocaps-org/updown-baseline/blob/master/scripts/build_vocabulary.py

import argparse
import json
import os

from nltk.tokenize import word_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Build a vocabulary out of nqa training file"
)

parser.add_argument(
    "-c",
    "--train-jsonpath",
    help="Path to NQA train json file.",
)

parser.add_argument("-t", "--word-count-threshold", type=int, default=20)

parser.add_argument(
    "-o",
    "--output-dirpath",
    help="Path to a (non-existent directory to save the vocabulary.",
)


# ------------------------------------------------------------------------------------------------
# All the punctuations in training data. We're removing them
PUNCTUATIONS = [
    "''", "'", "``", "`", "(", ")", "{", "}", ".", "?", "!", ",", ":", "-", "--", "...", ";"
]

# Special tokens which should be added (all, or a subset) to the vocabulary.
# Using a token for number because wiki is full of numbers
SPECIAL_TOKENS = ["@@UNKNOWN@@", "@@BOUNDARY@@", "@@NUMBER@@", "@@PADDING@@"]

def build_vocabulary(input_json, word_count_threshold):
    r"""
    Given a list of NQA examples, return a list of unique tokens thresholded
    by minimum occurence.
    """

    word_counts = {}

    # Accumulate unique tokens from all sequences.
    for item in tqdm(input_json):
        document_text = item["document_text"].lower().strip()
        document_tokens = word_tokenize(document_text)
        document_tokens = [ct for ct in document_tokens if ct not in PUNCTUATIONS]

        question_text = item["question_text"].lower().strip()
        question_tokens = word_tokenize(question_text)
        question_tokens = [ct for ct in question_tokens if ct not in PUNCTUATIONS]
        
        for token in document_tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
        
        for token in question_tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

    all_tokens = sorted(
        [key for key in word_counts if word_counts[key] >= word_count_threshold]
    )

    vocabulary = sorted(list(all_tokens))
    return vocabulary


if __name__ == "__main__":

    args = parser.parse_args()
    print("Loading annotations json from ", args.train_jsonpath)
    
    input_file = open(args.train_jsonpath)
    train_data = []
    for entry in input_file:
        train_data.append(json.loads(entry))

    print("Building vocabulary...")
    vocabulary = build_vocabulary(train_data, args.word_count_threshold)
    vocabulary = SPECIAL_TOKENS + vocabulary
    print("Vocabulary size (with special tokens): ", str(len(vocabulary)))

    # Write the vocabulary to separate namespace files in directory.
    print("Writing the vocabulary to", args.output_dirpath)

    os.makedirs(args.output_dirpath, exist_ok=True)

    with open(os.path.join(args.output_dirpath, "tokens.txt"), "w") as f:
        for token in vocabulary:
            f.write(token + "\n")