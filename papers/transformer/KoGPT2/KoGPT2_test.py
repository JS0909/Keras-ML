# !pip install ratsnlp

import torch
from ratsnlp.nlpbook.generation import GenerationTrainArguments
args = GenerationTrainArguments(
    pretrained_model_name="skt/kogpt2-base-v2",
    downstream_corpus_name="nsmc",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-generation",
    max_seq_length=32,
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    epochs=3,
    tpu_cores=0 if torch.cuda.is_available() else 8,
    seed=7,
)

from ratsnlp import nlpbook
nlpbook.set_seed(args)
nlpbook.set_logger(args)

from Korpora import Korpora
corpus = Korpora.load("modu_written", root_dir='') # 말뭉치 받은 디렉토리 설정

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    args.pretrained_model_name,
    eos_token="</s>",
)

from ratsnlp.nlpbook.generation import GenerationDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

train_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

val_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="test",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(
    args.pretrained_model_name
)

from ratsnlp.nlpbook.generation import GenerationTask
task = GenerationTask(model, args)
trainer = nlpbook.get_trainer(args)

trainer.fit(
    task,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)