import torch
import torch.nn as nn
import time
import json
import numpy as np
from transformers import BertTokenizerFast, BertForMaskedLM
from bert_model import NERModel
from transformers import RobertaTokenizerFast
from detect import detect_pii
import argparse
import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Detetion of personal information in text documents.')

    parser.add_argument('input_file', type=str,
                        help='the path to the JSON file containing the text and target')
    parser.add_argument('output_file_name', type=str,
                        help='the name for the output file')
    args = parser.parse_args()

    input_file = args.input_file
    out_file = args.output_file_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    ner_model = NERModel(model="roberta-base", num_labels=17)
    ner_model.load_state_dict(torch.load("model/3roberta_model.pt", map_location=device))

    with open(input_file, "r") as f:
        tmp = json.load(f)

    print(len(tmp))

    out = {}
    for document in tqdm.tqdm(tmp):
        out[document['doc_id']] = []

        # Privacy enhanced ER to detect all PII
        posList = detect_pii(document, ner_model, roberta_tokenizer)
        posList = list(posList.values())[0]

        out[document['doc_id']] = posList

    print(len(out))

    f_out = open(out_file, "w")
    json.dump(out, f_out, ensure_ascii=False)
