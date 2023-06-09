from argparse import ArgumentParser, Namespace
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import math
from pathlib import Path
from datetime import datetime

from model import CustomedModel

import wandb

from tqdm import trange

from sklearn.preprocessing import PowerTransformer

from typing import Optional

import os

COL_NAMES = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Album_type', 'Licensed', 'official_video', 'id', 'Track', 'Album', 'Uri', 'Url_spotify', 'Url_youtube', 'Comments', 'Description', 'Title', 'Channel', 'Composer', 'Artist']
ID = 'id'
LABEL = 'Danceability'
# TODO: Feature selection.
NUM_COL_NAMES = ['Energy','Key','Loudness','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Duration_ms','Views','Likes','Stream']
STR_COL_NAMES = ["Description", "Artist", "Composer", "Album", "Track"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # Data
    parser.add_argument(
        "--train_file",
        type=str,
        default="./data/train.csv",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./data/test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./models/",
    )
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--num_proc", type=int, default=2)
    
    # Training arguments
    parser.add_argument("--encoder_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=20)
    args = parser.parse_args()
    return args

def read_data(args):
    train_data = pd.read_csv(args.train_file, encoding = 'utf-8')
    test_data = pd.read_csv(args.test_file, encoding = 'utf-8')

    pt = PowerTransformer()
    # Replace all NaN with mean value of that column in training data
    for name in NUM_COL_NAMES:
        # Data transform
        # Normalization
        test_data[name] = (test_data[name]-train_data[name].mean())/train_data[name].std()
        train_data[name] = (train_data[name]-train_data[name].mean())/train_data[name].std()

        test_data[name] = test_data[name].fillna(0.)
        train_data[name] = train_data[name].fillna(0.)

        # Power Transform
        test_2d_arr = pt.fit_transform(test_data[name].to_numpy().reshape(-1,1))
        train_2d_arr = pt.fit_transform(train_data[name].to_numpy().reshape(-1,1))
        
        test_data[name] = pd.Series(test_2d_arr.reshape(-1))
        train_data[name] = pd.Series(train_2d_arr.reshape(-1))
    
    for name in STR_COL_NAMES:
        test_data[name] = test_data[name].fillna("nan")
        train_data[name] = train_data[name].fillna("nan")
    
    removed_col_names = list(set(COL_NAMES) - set(NUM_COL_NAMES) - set(STR_COL_NAMES) -set([LABEL]) - set([ID]))
    print(f"removed_col_names: {removed_col_names}")
    for name in removed_col_names:
        test_data.drop(name, axis=1, inplace=True)
        train_data.drop(name, axis=1, inplace=True)

    test_data[ID] = test_data[ID].astype("int32")
    return train_data, test_data
     
def preprocess_function(examples, tokenizer, max_length):
    # TODO: Test with different prompt.
    # text = examples["Description"]
    ############
    #  Prompt  #
    ############
    col_names = STR_COL_NAMES 
    text = ""
    for name in col_names:
        text += f'{name}: {examples[name]}. '
    
    processed_examples = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    processed_examples["text"] = text
    # Change this to real number
    if LABEL in examples:
        label = examples[LABEL]
        processed_examples["labels"] = float(label)
    
    processed_examples["related_features"] = [examples[name] for name in NUM_COL_NAMES]
    return processed_examples

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    # logits is a tuple
    logits = logits[0]
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)   
        logits = outputs["logits"][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        print(f"Saving model checkpoint to {output_dir}")

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model.encoder.save_pretrained(os.path.join(output_dir, "encoder"))
        torch.save(self.model.fc_layers.state_dict(), os.path.join(output_dir, "fc_layers.bin"))
        

def main(args):
    ##########
    #  Data  #
    ##########
    # train: num_rows: 15453
    # val: num_rows: 1717
    # test: num_rows: 6315
    # features: ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Album_type', 'Licensed', 'official_video', 'id', 'Track', 'Album', 'Uri', 'Url_spotify', 'Url_youtube', 'Comments', 'Description', 'Title', 'Channel', 'Composer', 'Artist']
    train_data, test_data = read_data(args)
    raw_train_ds, raw_test_ds = Dataset.from_pandas(train_data, split="train"), Dataset.from_pandas(test_data, split="test")
    
    raw_split_ds = raw_train_ds.train_test_split(test_size=0.1)
    
    ds = { "train": raw_split_ds["train"], 
            "validation": raw_split_ds["test"], 
            "test": raw_test_ds}
      
    ###########
    #  Model  #
    ###########
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
    
    if args.checkpoint is None:
        model = CustomedModel(encoder_name_or_path=args.encoder_name_or_path, num_classes=1, related_features_dim=len(NUM_COL_NAMES))
    else:
        print(f"Load from checkpoint: {args.checkpoint}")
        model = CustomedModel().from_pretrained(args.checkpoint, num_classes=1, related_features_dim=len(NUM_COL_NAMES))
    
    for split in ds:
        print(f"Mapping {split} split.")
        ds[split] = ds[split].map(
                    preprocess_function, 
                    num_proc=args.num_proc,
                    fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
                    )
              
    
    ###########
    #  Train  #
    ###########
    
    training_args = TrainingArguments(
        output_dir=args.repo_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epoch,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        weight_decay=0.01,
    )
    
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics_for_regression,
    )
    
    
    if not args.only_test:
        trainer.train()

    ####################
    #  Test Statistic  #
    ####################
    trainer.eval_dataset=ds["test"]
    trainer.evaluate()

    # Output prediction
    nb_batches = math.ceil(len(ds["test"])/args.batch)
    y_preds = []

    for i in trange(nb_batches):
        input_texts = ds["test"][i * args.batch: (i+1) * args.batch]["text"]
        # input_labels = raw_test_ds[i * args.batch: (i+1) * args.batch][LABEL]
        encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")
        tmp = model(**encoded, related_features=torch.tensor(ds["test"][i * args.batch: (i+1) * args.batch]['related_features']).to('cuda')).logits.reshape(-1).tolist()
        for idx in range(len(tmp)):
            if tmp[idx] > 9:
                tmp[idx] = 9
            if tmp[idx] < 0:
                tmp[idx] = 0
        y_preds += tmp

    df = pd.DataFrame([ds["test"][ID], y_preds], [ID, LABEL]).T
    df[ID] = df[ID].apply(round) 
    # df[LABEL] = df["Prediction"].apply(round) 
    # df.drop("Prediction", axis=1, inplace=True)
    df.to_csv(args.repo_dir / "pred.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    time = datetime.now().strftime('%Y%m%d-%H%M%S')
    args.repo_dir = args.output_dir / time


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="NTU-ML-Final",
        # track hyperparameters and run metadata
        config=args
    )

    main(args)