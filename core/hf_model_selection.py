import os, re
# os.environ["CUDA_VISIBLE_DEVICES"]="7" #,0,1,2,5,6"
# os.environ['HF_HOME']='/var/tellme/users/enguyen/hf_cache'
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch import nn
from torch.nn.functional import normalize
import evaluate
from core.model_selection import ModelSearchResult
from datasets import Dataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import math
import shutil
import logging

f1_metric = evaluate.load("f1")
text_column = 'text'
label_column = 'labels'


class BERTLike:
    """
    This is a wrapper class that enables use of BERT in the model_selection.select_model(). This is a higher level
    of abstraction than sklearn_HF and might be easier to use.
    """

    def __init__(self, model_name='bert-base-uncased', lr=1e-4, train_batch_size=8, eval_batch_size=8, num_epochs=1,
                 eval_steps=1, warmup_steps=0.1, max_length=64, max_steps=-1, save_model=False, output_dir=None):
        self.model_name = model_name
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_epochs = num_epochs
        self.eval_steps = eval_steps
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        self.max_steps = max_steps
        self.save_model = save_model
        self.output_dir = output_dir

    def fit(self, X_train, y_train, X_val, y_val):
        search_res_obj = train_using_val(X_train, y_train, X_val=X_val, y_val=y_val,
                                         model_name=self.model_name, lr=self.lr,
                                         train_batch_size=self.train_batch_size, eval_batch_size=self.eval_batch_size,
                                         num_epochs=self.num_epochs, eval_steps=self.eval_steps,
                                         warmup_steps=self.warmup_steps,
                                         max_length=self.max_length, max_steps=self.max_steps,
                                         save_model=self.save_model,
                                         output_dir=self.output_dir
                                         )
        if not self.save_model:
            shutil.rmtree(self.output_dir)
        self.model = search_res_obj.clf

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_cls_output(self, X, is_last_hidden_state=True, is_normalized=False):
        return self.model.get_cls_output(X, is_last_hidden_state=is_last_hidden_state, is_normalized=is_normalized)


class sklearn_HF:
    """
    This is a wrapper class over whatever train_using_val returns so it is usable by the main AL loop, i.e.,
    this is an object that has predict(), predict_proba().
    """
    def __init__(self, model, tokenizer, train_args, id2label, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = tokenizer.init_kwargs['max_length'] if 'max_length' in tokenizer.init_kwargs else None
        self.truncation = tokenizer.init_kwargs['truncation'] if 'truncation' in tokenizer.init_kwargs else False
        self.padding = tokenizer.init_kwargs['padding'] if 'padding' in tokenizer.init_kwargs else False
        self.train_args = train_args
        self.id2label = id2label
        self.output_dir = output_dir

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_ids = np.argmax(proba, axis=-1)
        predictions = [self.id2label[lid] for lid in pred_ids]
        return predictions

    def predict_proba(self, X):
        data = prepare_train_data(self.tokenizer, X, y=None, ds_type='test', label2id=None)
        trainer = Trainer(model=self.model, args=self.train_args)
        output = trainer.predict(data)
        proba = nn.functional.softmax(torch.tensor(output.predictions), dim=-1).numpy()
        return proba

    def get_cls_output(self, X, is_last_hidden_state=True, is_normalized=False):
        data = prepare_train_data(self.tokenizer, X, y=None, ds_type='test', label2id=None)
        data.set_format("torch")
        eval_dataloader = DataLoader(data, batch_size=self.train_args.per_device_eval_batch_size)
        cls_output = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: v.to(self.train_args.device) for k, v in batch.items()}
            with torch.no_grad():
                if is_last_hidden_state:
                    cls_output_batch = self.model.base_model(**batch).last_hidden_state[:,0,:]
                else:
                    cls_output_batch = self.model.base_model(**batch).pooler_output
            if cls_output is None:
                cls_output = cls_output_batch
            else:
                cls_output = torch.cat((cls_output, cls_output_batch))
        if is_normalized:
            cls_output = normalize(cls_output).detach().cpu().numpy()
        else:
            cls_output = cls_output.detach().cpu().numpy()

        return cls_output


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels, average="macro")

def tokenize_text(examples, tokenizer):
    return tokenizer(examples,
                     truncation=tokenizer.init_kwargs['truncation'],
                     padding=tokenizer.init_kwargs['padding'],
                     max_length=tokenizer.init_kwargs['max_length'],
                     )


def prepare_train_data(tokenizer, X, y=None, ds_type='train', label2id=None):
    """

    :param tokenizer:
    :param X: this is a numpy 1D array of strings
    :param y: numpy array of labels
    :param ds_type:
    :param label2id:
    :param max_length:
    :return:
    """
    if ds_type == 'train':
        # get train labels & IDs
        # convert to python type int otherwise there are problems with serialization
        unique_labels = list(map(int, np.unique(y)))
        id2label = dict(enumerate(unique_labels))
        label2id = dict([(v, k) for k, v in id2label.items()])

    if y is not None:
        # convert labels to ids
        y_ids = list(map(lambda t: label2id[t], y))
        # get HF dataset object
        data = Dataset.from_dict({text_column: X, label_column: y_ids})
    else:
        data = Dataset.from_dict({text_column: X})
    # tokenize data
    tokenize_func = partial(tokenize_text, tokenizer=tokenizer)
    data = data.map(lambda examples: tokenize_func(examples[text_column]), batched=True)
    data = data.remove_columns([text_column])

    if ds_type == 'train':
        return data, label2id, id2label
    else:
        return data


def train_using_val(X, y, val_size=0.2, X_val=None, y_val=None,
                    model_name='bert-base-uncased', lr=1e-4,
                    train_batch_size=8, eval_batch_size=8, num_epochs=1, eval_steps=1, warmup_steps=0.1,
                    max_length=64, max_steps=-1, save_model=False, output_dir=None, *args, **kwargs):
    """

    :param X: numpy 1D array of strings
    :param y:
    :param val_size: size of the validation set to be sampled with strat. from x, y
    :param X_val: you can also directly pass the validation data, this overrides val_size
    :param y_val:
    :param model_name:
    :param lr:
    :param batch_size:
    :param num_epochs:
    :param eval_steps:
    :param max_length:
    :param max_steps:
    :param save_model:
    :param output_dir:
    :return:
    """
    if X_val is None or y_val is None:
        logging.info(f"Validation data is not passed, we will use stratified random split of size={val_size}")
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, train_size=1 - val_size)
    else:
        X_train, y_train = X, y

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding='max_length', truncation=True, max_length=max_length)
    train_ds, label2id, id2label = prepare_train_data(tokenizer, X_train, y_train, ds_type='train',
                                                      label2id=None)
    dev_ds = prepare_train_data(tokenizer, X_val, y_val, ds_type='dev', label2id=label2id)

    data = DatasetDict({
        'train': train_ds,
        'dev': dev_ds
    })

    # Prepare base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    # train
    if type(eval_steps) == tuple and eval_steps[0] == 'num_evals_per_epoch':
        num_batches_per_epoch = math.ceil(len(y_train) / train_batch_size)
        eval_steps = max(1, min(int(num_batches_per_epoch / eval_steps[1]), 500))
        # eval_steps = 1/num_epochs/eval_steps[1]
    if type(warmup_steps) is float:
        warmup_steps = math.ceil(warmup_steps*math.ceil(len(y_train) / train_batch_size) * num_epochs)

    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_steps=max_steps,
        save_steps=eval_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        learning_rate=lr,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        logging_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        # greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    logging.info(
        f'Eval_steps: {trainer.args.eval_steps}, total_steps: {trainer.state.global_step}, warmup_steps: '
        f'{trainer.args.warmup_steps}')
    logging.info(f'Trainer state: {trainer.state}')

    if save_model:
        if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        # save model config and weights
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        model.config.to_json_file(os.path.join(output_dir, "model_config.json"))
        torch.save(train_args, os.path.join(output_dir, 'training_args.bin'))

    wrapper_obj = sklearn_HF(model, tokenizer, train_args, id2label, output_dir)
    return ModelSearchResult(wrapper_obj)


if __name__ == "__main__":
    output_dir = '../scratch/model_selector'
    train_file = '../scratch/SST2_train.tsv'
    df = pd.read_csv(train_file, sep='\t')
    X, y = df['sentence'].to_numpy(), df['label'].to_numpy()
    X, X_test, y, y_test = train_test_split(X, y, train_size=50, test_size=100, stratify=y)
    obj = train_using_val(X, y, max_steps=2, save_model=True, output_dir=output_dir)
    # print(obj.predict_proba(X_test))
    print(obj.predict(X_test))
