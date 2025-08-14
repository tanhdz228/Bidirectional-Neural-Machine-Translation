#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import time
import gc
import zipfile
from datetime import datetime
from typing import List, Optional
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        MarianMTModel,
        MarianTokenizer,
    )
except ImportError:
    print("Installing transformers...")
    os.system("pip install -U transformers")
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        MarianMTModel,
        MarianTokenizer,
    )

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'pretrained_inference_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    logger = logging.getLogger('PretrainedInference')
    logger.info(f"Log file created: {log_file}")
    return logger, log_file

class PretrainedMedicalTranslator:
    SUPPORTED_MODELS = {
        'helsinki-small': {
            'en2vi': 'Helsinki-NLP/opus-mt-en-vi',
            'vi2en': 'Helsinki-NLP/opus-mt-vi-en',
            'type': 'marian',
        },
        'vinai-envit5': {
            'model_name': 'vinai/envit5-translation',
            'type': 'mt5',
        },
        'nllb-1.3b': {
            'model_name': 'facebook/nllb-200-1.3B',
            'en_code': 'eng_Latn',
            'vi_code': 'vie_Latn',
            'type': 'nllb',
        },
        'nllb-3.3b': {
            'model_name': 'facebook/nllb-200-3.3B',
            'en_code': 'eng_Latn',
            'vi_code': 'vie_Latn',
            'type': 'nllb',
        },
    }
    
    def __init__(self, model_choice='helsinki-small', device='cuda', 
                 batch_size=8, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        if model_choice not in self.SUPPORTED_MODELS:
            self.logger.warning(f"Model {model_choice} not supported, defaulting to helsinki-small.")
            self.model_choice = 'helsinki-small'
        else:
            self.model_choice = model_choice
            
        self.model_config = self.SUPPORTED_MODELS[self.model_choice]
        self.model_type = self.model_config['type']
        
        self.logger.info(f"Initializing model: {self.model_choice} on device: {self.device} with batch size: {self.batch_size}")
        self._load_models()
    
    def _load_models(self):
        self.logger.info("Loading models...")
        start_time = time.time()
        
        try:
            use_fp16 = self.device.type == 'cuda'
            torch_dtype = torch.float16 if use_fp16 else torch.float32

            if self.model_type == 'marian':
                self.en2vi_model = MarianMTModel.from_pretrained(self.model_config['en2vi']).to(self.device).eval()
                self.en2vi_tokenizer = MarianTokenizer.from_pretrained(self.model_config['en2vi'])
                self.vi2en_model = MarianMTModel.from_pretrained(self.model_config['vi2en']).to(self.device).eval()
                self.vi2en_tokenizer = MarianTokenizer.from_pretrained(self.model_config['vi2en'])
            
            elif self.model_type in ['nllb', 'mt5']:
                model_name = self.model_config['model_name']
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True
                ).to(self.device).eval()
            
            load_time = time.time() - start_time
            self.logger.info(f"Models loaded in {load_time:.2f} seconds.")
            if self.device.type == 'cuda':
                self.logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
        except Exception as e:
            self.logger.error(f"Failed to load model '{self.model_choice}': {e}")
            raise
    
    def translate_batch(self, texts: List[str], direction='en2vi') -> List[str]:
        if not texts:
            return []
        
        translations = [""] * len(texts)
        try:
            with torch.no_grad():
                if self.model_type == 'marian':
                    model, tokenizer = (self.en2vi_model, self.en2vi_tokenizer) if direction == 'en2vi' else (self.vi2en_model, self.vi2en_tokenizer)
                    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    generated_tokens = model.generate(**inputs, max_length=512, num_beams=4)
                    translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    
                elif self.model_type == 'nllb':
                    src_lang, tgt_lang = (self.model_config['en_code'], self.model_config['vi_code']) if direction == 'en2vi' else (self.model_config['vi_code'], self.model_config['en_code'])
                    self.tokenizer.src_lang = src_lang
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                        max_length=512, num_beams=4
                    )
                    translations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    
                elif self.model_type == 'mt5':
                    prefix = "translate English to Vietnamese: " if direction == 'en2vi' else "translate Vietnamese to English: "
                    prefixed_texts = [f"{prefix}{text}" for text in texts]
                    inputs = self.tokenizer(prefixed_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    generated_tokens = self.model.generate(**inputs, max_length=512, num_beams=4)
                    translations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Batch translation error: {e}")
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return translations
    
    def translate_file(self, input_file: str, direction='en2vi', column_name: Optional[str] = None) -> List[str]:
        self.logger.info(f"Starting file translation: {input_file} ({direction})")
        if not os.path.exists(input_file):
            self.logger.error(f"File not found: {input_file}")
            return []
        
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            self.logger.error(f"Could not read CSV '{input_file}': {e}")
            return []

        if column_name is None:
            column_name = 'English' if 'English' in df.columns else 'Vietnamese' if 'Vietnamese' in df.columns else df.columns[0]
        
        self.logger.info(f"Translating {len(df)} rows from column '{column_name}'.")
        texts = df[column_name].fillna("").astype(str).tolist()
        
        translations = []
        with tqdm(total=len(texts), desc=f"Translating {direction}") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_translations = self.translate_batch(batch, direction)
                translations.extend(batch_translations)
                pbar.update(len(batch))
        
        self.logger.info(f"Finished translating {len(translations)} sentences.")
        return translations

def find_data_files(en_file='en.csv', vi_file='vi.csv'):
    for folder in ['.', 'data', '../data', '..']:
        en_path = Path(folder) / en_file
        vi_path = Path(folder) / vi_file
        if en_path.exists() and vi_path.exists():
            return str(en_path), str(vi_path)
    return None, None

def main():
    parser = argparse.ArgumentParser(description='Medical Translation using Pretrained Models')
    parser.add_argument('--model', type=str, default='helsinki-small', choices=list(PretrainedMedicalTranslator.SUPPORTED_MODELS.keys()), help='Pretrained model to use')
    parser.add_argument('--en-csv', type=str, default='en.csv', help='English CSV input file')
    parser.add_argument('--vi-csv', type=str, default='vi.csv', help='Vietnamese CSV input file')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--list-models', action='store_true', help='List all available models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable Pretrained Models for EN-VI Translation:")
        print("-" * 50)
        for model_id, config in PretrainedMedicalTranslator.SUPPORTED_MODELS.items():
            print(f"- {model_id:<15} (type: {config['type']})")
        print("-" * 50)
        sys.exit(0)
    
    logger, log_file = setup_logging()
    
    logger.info("="*60)
    logger.info("Starting Medical Translation Process")
    logger.info(f"Configuration -> Model: {args.model}, Device: {args.device}, Batch Size: {args.batch_size}")
    logger.info("="*60)
    
    en_path, vi_path = find_data_files(args.en_csv, args.vi_csv)
    if not en_path or not vi_path:
        logger.error(f"Could not find input files: {args.en_csv} and {args.vi_csv}. Searched in current, data/, and parent directories.")
        sys.exit(1)
    
    logger.info(f"Found English file: {en_path}")
    logger.info(f"Found Vietnamese file: {vi_path}")
    
    try:
        translator = PretrainedMedicalTranslator(
            model_choice=args.model,
            device=args.device,
            batch_size=args.batch_size,
            logger=logger
        )
        
        vi_translations = []
        if en_path:
            logger.info("--- Starting English to Vietnamese Translation ---")
            start_time = time.time()
            vi_translations = translator.translate_file(en_path, direction='en2vi', column_name='English')
            logger.info(f"EN->VI translation completed in {time.time() - start_time:.2f}s.")
        
        en_translations = []
        if vi_path:
            logger.info("--- Starting Vietnamese to English Translation ---")
            start_time = time.time()
            en_translations = translator.translate_file(vi_path, direction='vi2en', column_name='Vietnamese')
            logger.info(f"VI->EN translation completed in {time.time() - start_time:.2f}s.")

        logger.info("--- Processing and Saving Results ---")
        max_len = max(len(en_translations), len(vi_translations))
        if max_len == 0:
            logger.error("No translations were generated. Exiting.")
            sys.exit(1)
            
        results_df = pd.DataFrame({
            'English_From_Vietnamese': en_translations + [""] * (max_len - len(en_translations)),
            'Vietnamese_From_English': vi_translations + [""] * (max_len - len(vi_translations))
        })
        
        results_df.to_csv(args.output, index=False, encoding='utf-8')
        logger.info(f"Results saved to: {args.output}")
        
        zip_filename = 'results.zip'
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(args.output, arcname=os.path.basename(args.output))
        logger.info(f"Results also compressed to: {zip_filename}")
        
        logger.info("\n--- Final Summary ---")
        logger.info(f"Total EN->VI sentences processed: {len(vi_translations)}")
        logger.info(f"Total VI->EN sentences processed: {len(en_translations)}")
        logger.info(f"Log file available at: {log_file}")
        logger.info("✅ Process completed successfully!")
        
        del translator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()