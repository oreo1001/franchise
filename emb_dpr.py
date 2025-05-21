#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emb_dpr.py

LoRA 및 DPR 기법을 활용한 임베딩 모델 학습 스크립트.
- DPRDataset: question-context 데이터 로딩
- collate_fn: 배치 패딩 및 마스킹
- dpr_loss: DPR 손실 함수
- DPRModel: LoRA 적용된 질문/컨텍스트 인코더
- DPRTrainer: Trainer 커스터마이징
"""

import datetime
import os
import random
import json

import numpy as np
import torch
import torch.nn.functional as F
from numpy.core.multiarray import _reconstruct as np_reconstruct
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_utils import get_last_checkpoint

from peft import LoraConfig, get_peft_model

import wandb
from wandb import Settings

# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------
SAVE_DIR = "./checkpoints/"
MODEL_NAME = "./squad-v1/"
TRAIN_FILE = "./dataset/train.jsonl"
EVAL_FILE = "./dataset/eval.jsonl"
os.makedirs(SAVE_DIR, exist_ok=True)

wandb_api_key = os.environ.get("WANDB_API_KEY")

# -----------------------------------------------------------------------------
# 2) Dataset Definition
# -----------------------------------------------------------------------------
class DPRDataset(Dataset):
    """
    DPR 학습용 Dataset:
    - 각 예시는 question, positive_contexts, negative_contexts 로 구성
    - 토크나이저를 이용해 input_ids, attention_mask 반환
    """
    def __init__(self, path: str, tokenizer, max_len: int = 384):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = [json.loads(line) for line in open(path, 'r', encoding='utf-8')]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        ex = self.data[idx]
        q = self.tokenizer(ex["question"], truncation=True, max_length=self.max_len, return_tensors="pt")
        pos = [self.tokenizer(ctx["content"], truncation=True, max_length=self.max_len, return_tensors="pt")
               for ctx in ex["positive_contexts"]]
        neg = [self.tokenizer(ctx["content"], truncation=True, max_length=self.max_len, return_tensors="pt")
               for ctx in ex["negative_contexts"]]
        return {"q": q, "pos": pos, "neg": neg}


# -----------------------------------------------------------------------------
# 3) Collate Function
# -----------------------------------------------------------------------------
def collate_fn(batch: list) -> dict:
    """
    배치 내 질문과 컨텍스트(긍정/부정)를 패딩하고, pos_mask 생성
    """
    max_pos = max(len(x["pos"]) for x in batch)
    max_neg = max(len(x["neg"]) for x in batch)
    pad_id = tokenizer.pad_token_id

    qs_ids, qs_am = [], []
    ctx_ids, ctx_am = [], []
    pos_mask = []

    for ex in batch:
        qs_ids.append(ex["q"]["input_ids"].squeeze(0))
        qs_am.append(ex["q"]["attention_mask"].squeeze(0))

        p_ids = [p["input_ids"].squeeze(0) for p in ex["pos"]]
        p_am = [p["attention_mask"].squeeze(0) for p in ex["pos"]]
        for _ in range(max_pos - len(p_ids)):
            p_ids.append(torch.full((1, tokenizer.model_max_length), pad_id, dtype=torch.long))
            p_am.append(torch.zeros((1, tokenizer.model_max_length), dtype=torch.long))

        n_ids = [n["input_ids"].squeeze(0) for n in ex["neg"]]
        n_am = [n["attention_mask"].squeeze(0) for n in ex["neg"]]
        for _ in range(max_neg - len(n_ids)):
            n_ids.append(torch.full((1, tokenizer.model_max_length), pad_id, dtype=torch.long))
            n_am.append(torch.zeros((1, tokenizer.model_max_length), dtype=torch.long))

        ctx_ids.append(p_ids + n_ids)
        ctx_am.append(p_am + n_am)
        pos_mask.append([1.0] * len(p_ids) + [0.0] * (max_pos - len(p_ids) + max_neg))

    qs_batch = tokenizer.pad({"input_ids": qs_ids, "attention_mask": qs_am}, return_tensors="pt")

    B, C, _ = len(batch), max_pos + max_neg, tokenizer.model_max_length
    flat_ids = [item for pair in ctx_ids for item in pair]
    flat_am = [item for pair in ctx_am for item in pair]
    ctx_batch = tokenizer.pad({"input_ids": flat_ids, "attention_mask": flat_am}, return_tensors="pt")

    return {
        "q_ids": qs_batch["input_ids"],
        "q_attn": qs_batch["attention_mask"],
        "ctx_ids": ctx_batch["input_ids"].view(B, C, -1),
        "ctx_attn": ctx_batch["attention_mask"].view(B, C, -1),
        "pos_mask": torch.tensor(pos_mask, dtype=torch.float),
    }

# -----------------------------------------------------------------------------
# 4) DPR Loss Function
# -----------------------------------------------------------------------------
def dpr_loss(q_emb: torch.Tensor, ctx_emb: torch.Tensor, pos_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """DPR 손실: positive/all 비율 기반 log-loss"""
    scores = torch.matmul(q_emb.unsqueeze(1), F.normalize(ctx_emb, dim=-1).permute(0, 2, 1)).squeeze(1) / temperature
    valid = pos_mask.sum(dim=1) > 0
    scores, pos_mask = scores[valid], pos_mask[valid]
    if scores.numel() == 0:
        return torch.tensor(0.0, device=q_emb.device)
    exp_s = torch.exp(scores)
    return -torch.log((exp_s * pos_mask).sum(1) / exp_s.sum(1)).mean()


# -----------------------------------------------------------------------------
# 5) DPRModel Definition
# -----------------------------------------------------------------------------
class DPRModel(torch.nn.Module):
    """LoRA 적용된 DPR 모델"""
    def __init__(self, model_name: str):
        super().__init__()
        base_q, base_ctx = AutoModel.from_pretrained(model_name), AutoModel.from_pretrained(model_name)
        lora_cfg = LoraConfig(task_type="FEATURE_EXTRACTION", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query","key"])
        self.q_encoder, self.ctx_encoder = get_peft_model(base_q, lora_cfg), get_peft_model(base_ctx, lora_cfg)
        for enc in (self.q_encoder, self.ctx_encoder):
            enc.gradient_checkpointing_disable()
            enc.config.use_cache = False

    def forward(self, q_ids=None, q_attn=None, ctx_ids=None, ctx_attn=None, pos_mask=None, temperature: float = 1.0):
        q_out = self.q_encoder(input_ids=q_ids, attention_mask=q_attn)
        q_emb = F.normalize(q_out.last_hidden_state[:,0], dim=-1)
        B, C, L = ctx_ids.size()
        flat_ctx, flat_am = ctx_ids.view(B*C, L), ctx_attn.view(B*C, L)
        ctx_out = self.ctx_encoder(input_ids=flat_ctx, attention_mask=flat_am)
        ctx_emb = F.normalize(ctx_out.last_hidden_state[:,0], dim=-1).view(B, C, -1)
        exp_s = torch.exp(torch.matmul(q_emb.unsqueeze(1), ctx_emb.permute(0,2,1)).squeeze(1) / temperature)
        return ModelOutput(loss=-torch.log((exp_s * pos_mask).sum(1) / exp_s.sum(1)).mean())

    def save_pretrained(self, save_directory: str):
        self.q_encoder.save_pretrained(save_directory)


# -----------------------------------------------------------------------------
# 6) DPRTrainer Definition
# -----------------------------------------------------------------------------
class DPRTrainer(Trainer):
    """Trainer 커스터마이징: dataloader 및 loss override"""
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    def get_eval_dataloader(self, eval_dataset=None):
        return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = self._prepare_inputs(inputs)
        real = model.module if hasattr(model, 'module') else model
        q_out = real.q_encoder(input_ids=inputs['q_ids'], attention_mask=inputs['q_attn'])
        q_emb = F.normalize(q_out.last_hidden_state[:,0], dim=-1)
        B, C, L = inputs['ctx_ids'].size()
        flat_ctx, flat_am = inputs['ctx_ids'].view(B*C, L), inputs['ctx_attn'].view(B*C, L)
        ctx_out = real.ctx_encoder(input_ids=flat_ctx, attention_mask=flat_am)
        ctx_emb = F.normalize(ctx_out.last_hidden_state[:,0], dim=-1).view(B, C, -1)
        loss = dpr_loss(q_emb, ctx_emb, inputs['pos_mask'])
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

    def _load_rng_state(self, resume_from_checkpoint: str):
        if not resume_from_checkpoint: return
        rng_file = os.path.join(resume_from_checkpoint, 'rng_state.pth')
        if os.path.isfile(rng_file):
            state = torch.load(rng_file, weights_only=False)
            random.setstate(state['python'])
            torch.set_rng_state(state['torch'])
            if 'cuda' in state: torch.cuda.set_rng_state_all(state['cuda'])
            if 'numpy' in state: np.random.set_state(state['numpy'])


# -----------------------------------------------------------------------------
# 7) Main: Initialization & Training
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DPRModel(MODEL_NAME)

    train_ds = DPRDataset(TRAIN_FILE, tokenizer)
    eval_ds = DPRDataset(EVAL_FILE, tokenizer)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-dpr-stal-v2"
    
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        learning_rate=5e-6,
        warmup_steps=2000,
        weight_decay=0.01,
        num_train_epochs=3,
        fp16=True,
        logging_steps=100,
        eval_strategy='steps', eval_steps=1000,
        save_strategy='steps', save_steps=1000, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        lr_scheduler_type='linear', max_grad_norm=1.0,
        report_to='wandb', run_name=run_name
    )

    # W&B 및 RNG 복원 설정
    torch.serialization.add_safe_globals([np_reconstruct, np.generic, np.ndarray, np.dtype])
    wandb.login(key=wandb_api_key)
    wandb.init(entity=wandb_api_key, project="FTC-Competition", name=training_args.run_name, config=training_args.to_dict(), settings=Settings(init_timeout=180))

    trainer = DPRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    last_ckpt = get_last_checkpoint(SAVE_DIR)
    if last_ckpt and os.path.isfile(os.path.join(last_ckpt, 'trainer_state.json')):
        print(f"▶️ 마지막 체크포인트 발견: {last_ckpt} 에서 재개합니다.")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("▶️ 체크포인트를 찾지 못해 처음부터 학습을 시작합니다.")
        trainer.train()

    trainer.save_model(SAVE_DIR)
    model.q_encoder.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"▶️ Training 완료! 모델과 토크나이저 저장 위치: {SAVE_DIR}")