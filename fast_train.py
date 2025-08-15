import os
import gc
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sentencepiece as spm
from underthesea import word_tokenize
from sacrebleu import BLEU
from tqdm import tqdm
import pandas as pd
import numpy as np

from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    MT5ForConditionalGeneration, MT5Tokenizer,
    NllbTokenizer, AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)

from dataset import MedicalTranslationDataset, collate_fn
from preprocessor import MedicalNMTPreprocessor

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'fast_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), log_file

class PretrainedMedicalNMT(nn.Module):
    def __init__(self, model_type='mbart-large', device='cuda'):
        super().__init__()
        self.model_type = model_type
        self.device = device
        
        if model_type == 'mbart-large':
            self.model = MBartForConditionalGeneration.from_pretrained(
                'facebook/mbart-large-50-many-to-many-mmt',
                torch_dtype=torch.float16
            )
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                'facebook/mbart-large-50-many-to-many-mmt'
            )
            self.en_code = "en_XX"
            self.vi_code = "vi_VN"
            
        elif model_type == 'mt5-base':
            self.model = MT5ForConditionalGeneration.from_pretrained(
                'google/mt5-base',
                torch_dtype=torch.float16
            )
            self.tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')
            
        elif model_type == 'nllb-distilled':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                'facebook/nllb-200-distilled-600M',
                torch_dtype=torch.float16
            )
            self.tokenizer = NllbTokenizer.from_pretrained(
                'facebook/nllb-200-distilled-600M'
            )
            self.en_code = "eng_Latn"
            self.vi_code = "vie_Latn"
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.add_medical_adapters()
        
    def add_medical_adapters(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
        
        for name, module in base_model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'q_proj'):
                in_features = module.q_proj.in_features
                out_features = module.q_proj.out_features
                rank = 16
                
                module.adapter_down = nn.Linear(in_features, rank, bias=False)
                module.adapter_up = nn.Linear(rank, out_features, bias=False)
                
                nn.init.kaiming_uniform_(module.adapter_down.weight)
                nn.init.zeros_(module.adapter_up.weight)
                
                module.adapter_down.weight.requires_grad = True
                module.adapter_up.weight.requires_grad = True
                module.adapter_down.to(self.device)
                module.adapter_up.to(self.device)
                
                original_forward = module.forward
                def adapted_forward(self, *args, **kwargs):
                    forward_output = original_forward(*args, **kwargs)

                    if isinstance(forward_output, tuple):
                        output_tensor = forward_output[0]
                    else:
                        output_tensor = forward_output

                    if hasattr(self, 'adapter_down'):
                        hidden_states = kwargs.get('hidden_states')
                        if hidden_states is None:
                            if args:
                                hidden_states = args[0]
                            else:
                                return forward_output

                        adapter_out = self.adapter_up(self.adapter_down(hidden_states))
                        output_tensor = output_tensor + adapter_out

                    if isinstance(forward_output, tuple):
                        return (output_tensor,) + forward_output[1:]
                    else:
                        return output_tensor

                module.forward = adapted_forward.__get__(module, module.__class__)
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask, **kwargs):
        kwargs.setdefault('max_length', 256)
        kwargs.setdefault('num_beams', 4)
        kwargs.setdefault('early_stopping', True)
        
        kwargs['forced_bos_token_id'] = self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

class FastMedicalDataset(Dataset):
    def __init__(self, en_file, vi_file, tokenizer, model_type, max_length=256, max_samples=None):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length
        self.data = []
        
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(vi_file, 'r', encoding='utf-8') as f_vi:
            
            en_lines = f_en.readlines()
            vi_lines = f_vi.readlines()
            
            if max_samples:
                en_lines = en_lines[:max_samples]
                vi_lines = vi_lines[:max_samples]
            
            for en, vi in tqdm(zip(en_lines, vi_lines), desc="Loading data"):
                en = en.strip()
                vi = vi.strip()
                if en and vi:
                    self.data.append({'en': en, 'vi': vi})
        
        print(f"Loaded {len(self.data)} sentence pairs")
    
    def __len__(self):
        return len(self.data) * 2
    
    def __getitem__(self, idx):
        pair_idx = idx // 2
        direction = 'en2vi' if idx % 2 == 0 else 'vi2en'

        pair = self.data[pair_idx]

        if direction == 'en2vi':
            source, target = pair['en'], pair['vi']
            if self.model_type == 'mbart-large':
                self.tokenizer.src_lang = "en_XX"
                self.tokenizer.tgt_lang = "vi_VN"
            elif self.model_type == 'nllb-distilled':
                self.tokenizer.src_lang = "eng_Latn"
                self.tokenizer.tgt_lang = "vie_Latn"
        else:
            source, target = pair['vi'], pair['en']
            if self.model_type == 'mbart-large':
                self.tokenizer.src_lang = "vi_VN"
                self.tokenizer.tgt_lang = "en_XX"
            elif self.model_type == 'nllb-distilled':
                self.tokenizer.src_lang = "vie_Latn"
                self.tokenizer.tgt_lang = "eng_Latn"

        if self.model_type == 'mt5-base':
            if direction == 'en2vi':
                source = f"translate English to Vietnamese: {source}"
            else:
                source = f"translate Vietnamese to English: {source}"

        model_inputs = self.tokenizer(
            source,
            text_target=target,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': model_inputs['labels'].squeeze(),
            'direction': direction
        }

def collate_pretrained(batch):
    en2vi = [item for item in batch if item['direction'] == 'en2vi']
    vi2en = [item for item in batch if item['direction'] == 'vi2en']
    
    batches = []
    for direction_batch in [en2vi, vi2en]:
        if not direction_batch:
            continue
        
        input_ids = pad_sequence(
            [item['input_ids'] for item in direction_batch],
            batch_first=True,
            padding_value=0
        )
        attention_mask = pad_sequence(
            [item['attention_mask'] for item in direction_batch],
            batch_first=True,
            padding_value=0
        )
        labels = pad_sequence(
            [item['labels'] for item in direction_batch],
            batch_first=True,
            padding_value=-100
        )
        
        batches.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'direction': direction_batch[0]['direction']
        })
    
    return batches

class FastTrainer:
    def __init__(self, model, train_loader, val_loader, args, logger, num_gpus=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.logger = logger
        self.num_gpus = num_gpus
        
        if num_gpus > 1:
            self.logger.info(f"Using {num_gpus} GPUs with DataParallel")
            self.model = DataParallel(model)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Trainable params: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M")
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.01
        )
        
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.steps_per_epoch * args.epochs
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(500, self.total_steps // 10),
            num_training_steps=self.total_steps
        )
        
        self.scaler = GradScaler()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.start_time = time.time()
        self.target_hours = 6
    
    def train_step(self, batch_list):
        total_loss = 0
        
        self.optimizer.zero_grad()
        
        for batch in batch_list:
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            
            with autocast():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                
                if self.num_gpus > 1:
                    loss = loss.mean()
            
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            1.0
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        self.global_step += 1
        
        return total_loss / len(batch_list)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_list in self.val_loader:
                for batch in batch_list:
                    batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                    
                    with autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                    
                    loss = outputs.loss
                    if self.num_gpus > 1:
                        loss = loss.mean()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                if num_batches >= 50:
                    break
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def train(self):
        self.logger.info(f"Starting fast training for {self.args.epochs} epochs")
        self.logger.info(f"Target time: {self.target_hours} hours")
        self.logger.info(f"Steps per epoch: {self.steps_per_epoch}")
        
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            epoch_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            
            for batch_list in pbar:
                loss = self.train_step(batch_list)
                epoch_loss += loss
                
                elapsed = (time.time() - self.start_time) / 3600
                eta = (elapsed / (self.global_step + 1)) * self.total_steps
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}',
                    'elapsed': f'{elapsed:.1f}h',
                    'eta': f'{eta:.1f}h'
                })
                
                if self.global_step % 2000 == 0:
                    val_loss = self.validate()
                    self.logger.info(f"Step {self.global_step}: val_loss={val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_model()
            
            epoch_time = (time.time() - epoch_start) / 60
            avg_loss = epoch_loss / len(self.train_loader)
            
            val_loss = self.validate()
            
            self.logger.info(f"Epoch {epoch+1} - Time: {epoch_time:.1f}min, "
                           f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()
            
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.target_hours:
                self.logger.info(f"Reached time limit ({elapsed_hours:.1f}h)")
                break
        
        total_time = (time.time() - self.start_time) / 3600
        self.logger.info(f"Training completed in {total_time:.1f} hours")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_model(self):
        save_path = os.path.join(self.args.checkpoint_dir, 'best_model_fast.pt')
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'tokenizer': model_to_save.tokenizer,
            'model_type': self.args.model_type,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }, save_path)
        
        self.logger.info(f"Best model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Fast Medical NMT Training (~6 hours)')
    
    parser.add_argument('--model-type', type=str, default='nllb-distilled',
                       choices=['mbart-large', 'mt5-base', 'nllb-distilled'],
                       help='Pretrained model to use')
    
    parser.add_argument('--train-en', type=str, default='data/train.en.txt')
    parser.add_argument('--train-vi', type=str, default='data/train.vi.txt')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit training samples for faster training')
    
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs (3 for 6-hour training)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=256)
    
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    logger, log_file = setup_logging('logs')
    logger.info("="*60)
    logger.info("Fast Medical NMT Training")
    logger.info("="*60)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"GPUs: {args.num_gpus}")
    logger.info(f"Log file: {log_file}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        args.num_gpus = min(args.num_gpus, gpu_count)
    else:
        logger.warning("No GPU available, using CPU")
        args.num_gpus = 0
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logger.info(f"Loading pretrained model: {args.model_type}")
    model = PretrainedMedicalNMT(args.model_type)
    
    logger.info("Loading datasets...")
    train_dataset = FastMedicalDataset(
        args.train_en, args.train_vi,
        model.tokenizer, args.model_type,
        max_length=args.max_length,
        max_samples=args.max_samples
    )
    
    val_dataset = FastMedicalDataset(
        args.train_en, args.train_vi,
        model.tokenizer, args.model_type,
        max_length=args.max_length,
        max_samples=2000
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size * max(args.num_gpus, 1),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pretrained,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * max(args.num_gpus, 1) * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pretrained,
        pin_memory=True
    )
    
    trainer = FastTrainer(
        model, train_loader, val_loader,
        args, logger, num_gpus=args.num_gpus
    )
    
    trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Best model saved at: {args.checkpoint_dir}/best_model_fast.pt")

if __name__ == "__main__":
    main()