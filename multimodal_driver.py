from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import pickle
import logging
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, DebertaTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from bert import MAG_BertForSequenceClassification
from xlnet import MAG_XLNetForSequenceClassification
from roberta import MAG_RobertaForSequenceClassification
from deberta import MAG_DebertaForSequenceClassification

from argparse_utils import str2bool, seed
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=40)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=300)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base", "deberta-base"],
    default="deberta-base",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")

pwd = os.path.abspath(__file__)
logging.basicConfig(
            filename=os.path.join(Path(pwd).parent, "model.log"),
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %p %I:%M:%S',
            level=logging.INFO
        )
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("trainLogger")
logger.addHandler(logging.StreamHandler(sys.stdout))

args = parser.parse_args()


def return_unk():
    return 0

"""
필요시, 아래와 같은 InputFeatures class 생성
roberta 같은 경우 token_type_ids를 사용하지 않기 때문에 segment id 를 제외
"""


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputFeatures_roberta(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, attention_mask, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.attention_mask = attention_mask
        self.label_id = label_id


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


"""
본인이 선택한 모델에 맞게 필요시 아래 코드 수정
"""

def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        if args.model == "bert-base-uncased" or args.model == "deberta-base":
            prepare_input = prepare_bert_input
        elif args.model == "xlnet-base-cased":
            prepare_input = prepare_xlnet_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def convert_to_features_roberta(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        prepare_input = prepare_roberta_input
        input_ids, visual, acoustic, attention_mask = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(attention_mask) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures_roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )

    return features


"""
본인이 선택한 모델의 맞게 인풋 preparation 코드짜기
"""


def prepare_bert_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def prepare_xlnet_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    # PAD special tokens
    tokens = tokens + [SEP] + [CLS]
    audio_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, audio_zero, audio_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual, visual_zero, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens) - 1) + [2]

    pad_length = (args.max_seq_length - len(segment_ids))

    # then zero pad the visual and acoustic
    audio_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((audio_padding, acoustic))

    video_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((video_padding, visual))

    input_ids = [PAD_ID] * pad_length + input_ids
    input_mask = [0] * pad_length + input_mask
    segment_ids = [3] * pad_length + segment_ids

    return input_ids, visual, acoustic, input_mask, segment_ids


def prepare_roberta_input(tokens, visual, acoustic, tokenizer):
    # print(tokens)
    PAD_ID = tokenizer.pad_token_id

    # Pad zero vectors for acoustic / visual vectors to account for <s> / </s> tokens
    audio_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((audio_zero, acoustic, audio_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [0] + input_ids + [2]
    attention_mask = [1] * len(input_ids)
    # print(len(input_ids))

    pad_length = (args.max_seq_length - len(input_ids))

    # then zero pad the visual and acoustic
    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    input_ids = input_ids + [PAD_ID] * pad_length
    attention_mask = attention_mask + [0] * pad_length

    return input_ids, visual, acoustic, attention_mask


"""
본인이 선택한 pretrained 된 tokenizer 생성
"""

def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model)
    elif model == "xlnet-base-cased":
        return XLNetTokenizer.from_pretrained(model)
    elif model == "roberta-base":
        return RobertaTokenizer.from_pretrained(model)
    elif model == "deberta-base":
        return DebertaTokenizer.from_pretrained('microsoft/' + model)
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'xlnet-base-cased or 'roberta-base' or 'deberta-base', but received {}".format(
                model
            )
        )


"""
본인이 선택한 모델 추가
"""

def get_appropriate_dataset(data):

    tokenizer = get_tokenizer(args.model)

    if args.model == "bert-base-uncased" or args.model == "xlnet-base-cased" or args.model == "deberta-base":
        features = convert_to_features(data, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long)
        all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
        all_acoustic = torch.tensor(
            [f.acoustic for f in features], dtype=torch.float)
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(
            all_input_ids,
            all_visual,
            all_acoustic,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
        )

    elif args.model == "roberta-base":
        features = convert_to_features_roberta(data, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long)
        all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
        all_acoustic = torch.tensor(
            [f.acoustic for f in features], dtype=torch.float)
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(
            all_input_ids,
            all_visual,
            all_acoustic,
            all_attention_mask,
            all_label_ids,
        )

    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
본인 모델 추가
"""

def prep_for_training(num_train_optimization_steps: int):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    if args.model == "bert-base-uncased":
        model = MAG_BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1,
        )
    elif args.model == "xlnet-base-cased":
        model = MAG_XLNetForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1
        )
    elif args.model == "roberta-base":
        model = MAG_RobertaForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1
        )
    elif args.model == "deberta-base":
        model = MAG_DebertaForSequenceClassification.from_pretrained(
            'microsoft/' + args.model, multimodal_config=multimodal_config, num_labels=1
        )

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps,
        num_training_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, optimizer, scheduler


"""
본인 모델 train/eval/test에 추가
"""

def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        if args.model == "bert-base-uncased" or args.model == "xlnet-base-cased" or args.model == "deberta-base":
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
        elif args.model == "roberta-base":
            input_ids, visual, acoustic, attention_mask, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                attention_mask=attention_mask,
                labels=None,
            )
        logits = outputs[0]
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            if args.model == "bert-base-uncased" or args.model == "xlnet-base-cased" or args.model == "deberta-base":
                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                outputs = model(
                    input_ids,
                    visual,
                    acoustic,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            elif args.model == "roberta-base":
                input_ids, visual, acoustic, attention_mask, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                outputs = model(
                    input_ids,
                    visual,
                    acoustic,
                    attention_mask=attention_mask,
                    labels=None,
                )
            logits = outputs[0]

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            if args.model == "bert-base-uncased" or args.model == "xlnet-base-cased" or args.model == "deberta-base":
                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                outputs = model(
                    input_ids,
                    visual,
                    acoustic,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            elif args.model == "roberta-base":
                input_ids, visual, acoustic, attention_mask, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                outputs = model(
                    input_ids,
                    visual,
                    acoustic,
                    attention_mask=attention_mask,
                    labels=None,
                )

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    ##### binary #####
    preds = preds >= 0
    y_test = y_test >= 0

    ##### 7 class #####
    # preds = preds.astype(int)
    # y_test = y_test.astype(int)

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, mae, corr, f_score


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_accuracies = []

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        test_acc, test_mae, test_corr, test_f_score = test_score_model(
            model, test_data_loader
        )

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc:{}, test_mae:{}, test_corr:{}, test_f_score:{}".format(
                epoch_i, train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score
            )
        )
        logger.info(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc:{}, test_mae:{}, test_corr:{}, test_f_score:{}".format(
                epoch_i, train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score
            )
        )

        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)
        print("best_test_acc:{}".format(max(test_accuracies)))
        logger.info("best_test_acc:{}".format(max(test_accuracies)))

def main():
    set_random_seed(args.seed)
    logger.info("seed:{}".format(args.seed))

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
