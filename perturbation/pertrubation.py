from __future__ import annotations
from collections import deque
import numpy as np
import intervaltree
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForMaskedLM, RobertaForMaskedLM, RobertaTokenizerFast
import math
import json
import pandas as pd
import csv
import tqdm

class mlmbert:
    def __init__(self, device, tokenizer, model, max_segment_size = 512, N = 3):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.to(device)
        self.max_segment_size = max_segment_size
        self.N = N

    def get_model_predictions(self, input_ids, attention_mask):
        """Given tokenized input identifiers and an associated attention mask (where the
        tokens to predict have a mask value set to 0), runs the BERT language and returns
        the (unnormalized) prediction scores for each token.

        If the input length is longer than max_segment size, we split the document in
        small segments, and then concatenate the model predictions for each segment."""

        nb_tokens = len(input_ids)

        input_ids = torch.tensor(input_ids)[None, :].to(self.device)
        attention_mask = torch.tensor(attention_mask)[None, :].to(self.device)

        # If the number of tokens is too large, we split in segments
        if nb_tokens > self.max_segment_size:
            nb_segments = math.ceil(nb_tokens / self.max_segment_size)

            # Split the input_ids (and add padding if necessary)
            split_pos = [self.max_segment_size * (i + 1) for i in range(nb_segments - 1)]
            input_ids_splits = torch.tensor_split(input_ids[0], split_pos)
            # input_ids_splits = torch.tensor_split(input_ids[0], nb_segments)
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_splits, batch_first=True)

            # Split the attention masks
            attention_mask_splits = torch.tensor_split(attention_mask[0], split_pos)
            # attention_mask_splits = torch.tensor_split(attention_mask[0], nb_segments)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_splits, batch_first=True)

        # Run the model on the tokenized inputs + attention mask
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # And get the resulting prediction scores
        scores = outputs.logits

        # If the batch contains several segments, concatenate the result
        if len(scores) > 1:
            scores = torch.vstack([scores[i] for i in range(len(scores))])
            scores = scores[:nb_tokens]
        else:
            scores = scores[0]
        return scores

    def get_proba(self, probs_actual, text_spans, tokens_by_span):
        """
        :param probs_actual: (L,) The proba for each token
        :param text_spans:
        :param token_by_span:
        """
        res = []
        for text_span in text_spans:
            ## If the span does not include any actual token, skip
            ## Normally will not happen
            if not tokens_by_span[text_span]:
                continue
            # Added 1e-60 to avoid error
            res.append(sum([np.log10(probs_actual[token_idx]+1e-60) for token_idx in tokens_by_span[text_span]]))
            #res.append(np.prod([probs_actual[token_idx] for token_idx in tokens_by_span[text_span]]))
        return res

    def get_tokens_by_span(self, bert_token_spans, text_spans):
        """Given two lists of spans (one with the spans of the BERT tokens, and one with
        the text spans), returns a dictionary where each text span is associated
        with the indices of the BERT tokens it corresponds to."""

        # We create an interval tree to facilitate the mapping
        text_spans_tree = intervaltree.IntervalTree()

        for start, end in text_spans:
            text_spans_tree[start:end] = True

        # We create the actual mapping between spans and tokens index in the token list
        tokens_by_span = {span: [] for span in text_spans}
        for token_idx, (start, end) in enumerate(bert_token_spans):
            for span_start, span_end, _ in text_spans_tree[start:end]:
                tokens_by_span[(span_start, span_end)].append(token_idx)

        # And control that everything is correct
        for span_start, span_end in text_spans:
            if len(tokens_by_span[(span_start, span_end)]) == 0:
                print("Warning: span (%i,%i) without any token" % (span_start, span_end))
        return tokens_by_span


    def get_probability(self, text, text_spans):
        """
        Input: text,text_spans (entity position in the annotation)
        Output: blacklist for each entity
        """
        blacklist = []

        tokenized_res = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = tokenized_res["input_ids"]
        input_ids_copy = np.array(input_ids)
        bert_token_spans = tokenized_res['offset_mapping']
        tokens_by_span = self.get_tokens_by_span(bert_token_spans, text_spans)

        attention_mask = tokenized_res["attention_mask"]
        for token_indices in tokens_by_span.values():
            for token_idx in token_indices:
                attention_mask[token_idx] = 0
                input_ids[token_idx] = self.tokenizer.mask_token_id

        # print(self.tokenizer.mask_token_id)
        # print('Ids:', input_ids)

        logits = self.get_model_predictions(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1) # (L, Number of tokens in the dict)

        # Get prob for the input tokens
        probs_actual = probs[torch.arange(len(input_ids_copy)), input_ids_copy]  # (L,)
        probs_actual = probs_actual.detach().cpu().numpy()

        logproba = self.get_proba(probs_actual, text_spans, tokens_by_span)

        return logproba


if __name__ == "__main__":

    max_segment_size = 512
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    model = RobertaForMaskedLM.from_pretrained('roberta-large')
    N = 3

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mlmbert_model = mlmbert(device, tokenizer, model)

    header = ['doc_id', 'target_span', 'target_text', 'target_full_prob', 'comp_spans', 'target_prob_without', 'difference']
    with open('data/gold/wiki_test.json', 'r') as f, open('wiki_test_output.csv', 'w', newline='') as f_output:
        f = json.load(f)
        csv_output = csv.DictWriter(f_output, fieldnames=header, restval='NA')
        csv_output.writeheader()

        out = []

        for i in tqdm.tqdm(f):
            same_d = {}
            d = {}
            text = i['text']
            id_ = i['doc_id']
            text_spans = []
            for annotator in i['annotations']:
                for annotation in i['annotations'][annotator]['entity_mentions']:
                    if (annotation['start_offset'], annotation['end_offset']) not in text_spans:
                        text_spans.append((annotation['start_offset'], annotation['end_offset']))
                        if annotation['entity_id'] not in same_d:
                            same_d[annotation['entity_id']] = []
                            same_d[annotation['entity_id']].append((annotation['start_offset'], annotation['end_offset']))
                        else:
                            same_d[annotation['entity_id']].append((annotation['start_offset'], annotation['end_offset']))

            new_spans = []
            for span in text_spans:
                for k,v in same_d.items():
                    if span in v:
                        if span == v[0]:
                            new_spans.append(v)
                        else:
                            new_l = [i for i in v if i!=span]
                            new_l.insert(0, span)
                            new_spans.append(new_l)          


            for span in tqdm.tqdm(new_spans): ##spans with same id, first element is the target, multiple length lists
                lst = []
                d = {}
                probability_full = mlmbert_model.get_probability(text, span)[0]
                target = []
                target.append(span)
                new_list = [x for x in new_spans if x != span]
                # print(new_list)
                for el in new_list:
                    # print(el)
                    d['doc_id'] = id_
                    d['target_span'] = span[0]
                    d['target_text'] = text[span[0][0]:span[0][1]]
                    d['target_full_prob'] = probability_full
                    d['comp_spans'] = el
                    target.append(el)
                    target_ = [item for sublist in target for item in sublist]

                    probability = mlmbert_model.get_probability(text, target_)
                    d['target_prob_without'] = probability[0]
                    d['difference'] = probability_full - probability[0]
                    csv_output.writerow(d)
                    target.remove(el)
                target = []
                d = {}

        print('\nThis script is done now.\n')





