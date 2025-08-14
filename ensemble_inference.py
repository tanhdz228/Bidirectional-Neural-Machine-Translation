#!/usr/bin/env python3

import os
import sys
import torch
import pandas as pd
import logging
import time
import zipfile
import argparse
from typing import List
from collections import Counter
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("Required libraries not found. Installing 'transformers' and 'sentencepiece'...")
    os.system("pip install -U transformers sentencepiece")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'ensemble_inference.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class EnsembleTranslator:
    def __init__(self, device='cuda'):
        self.logger = setup_logging()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}

        self.models_config = {
            'nllb-3.3b': {
                'weight': 0.4,
                'model_name': 'facebook/nllb-200-3.3B',
                'type': 'nllb'
            },
            'nllb-1.3b': {
                'weight': 0.3,
                'model_name': 'facebook/nllb-200-1.3B',
                'type': 'nllb'
            },
            'vinai-envit5': {
                'weight': 0.3,
                'model_name': 'vinai/envit5-translation',
                'type': 'envit5'
            }
        }
        
        self.load_models()

    def load_models(self):
        self.logger.info(f"Initializing models on device: {self.device}")
        
        for model_id, config in self.models_config.items():
            self.logger.info(f"Loading {model_id}...")
            
            try:
                if '3.3B' in config['model_name'] and self.device.type == 'cuda':
                    if torch.cuda.get_device_properties(0).total_memory < 10 * 1e9:
                        self.logger.warning(f"Skipping {model_id} due to insufficient GPU memory (< 10GB).")
                        continue
                
                tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    config['model_name'],
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device).eval()
                
                self.models[model_id] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'type': config['type'],
                    'weight': config['weight']
                }
                self.logger.info(f"Successfully loaded {model_id}.")

            except Exception as e:
                self.logger.error(f"Could not load {model_id}. Reason: {e}")
        
        total_weight = sum(m['weight'] for m in self.models.values())
        if total_weight > 0:
            for model_id in self.models:
                self.models[model_id]['weight'] /= total_weight
        
        self.logger.info(f"Ensemble ready with {len(self.models)} active models.")

    def translate_single_model(self, text: str, model_id: str, direction: str) -> str:
        model_info = self.models[model_id]
        
        try:
            with torch.no_grad():
                tokenizer = model_info['tokenizer']
                model = model_info['model']
                
                if model_info['type'] == 'nllb':
                    src_lang, tgt_lang = ("eng_Latn", "vie_Latn") if direction == 'en2vi' else ("vie_Latn", "eng_Latn")
                    tokenizer.src_lang = src_lang
                    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                    generated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                        max_length=512,
                        num_beams=4
                    )
                elif model_info['type'] == 'envit5':
                    prompt = f"en: {text}" if direction == 'en2vi' else f"vi: {text}"
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                    generated_tokens = model.generate(**inputs, max_length=512, num_beams=4)

                return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                    
        except Exception as e:
            self.logger.error(f"Translation failed for {model_id} on text: '{text[:30]}...'. Error: {e}")
            return ""

    def ensemble_translate(self, text: str, direction: str) -> str:
        if not text or not text.strip():
            return ""
        
        translations = {model_id: self.translate_single_model(text, model_id, direction) for model_id in self.models}
        
        # Default to the highest weighted model's translation
        best_model_id = max(self.models, key=lambda m_id: self.models[m_id]['weight'])
        return translations.get(best_model_id, "")

    def translate_file(self, input_file: str, direction: str) -> List[str]:
        self.logger.info(f"Processing file {input_file} for {direction} translation.")
        
        try:
            df = pd.read_csv(input_file)
            texts = df[df.columns[0]].fillna("").astype(str).tolist()
        except FileNotFoundError:
            self.logger.error(f"Input file not found: {input_file}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to read {input_file}. Error: {e}")
            return []

        translations = []
        for text in tqdm(texts, desc=f"Translating {direction}"):
            translations.append(self.ensemble_translate(text, direction))
        
        return translations

def main():
    parser = argparse.ArgumentParser(description="Ensemble translation using NLLB and enViT5 models.")
    parser.add_argument('--en-csv', default='en.csv', help='Path to the English source CSV file.')
    parser.add_argument('--vi-csv', default='vi.csv', help='Path to the Vietnamese source CSV file.')
    parser.add_argument('--output', default='ensemble_results.csv', help='Name for the output CSV file.')
    parser.add_argument('--device', default='cuda', help='Device to run on (cuda or cpu).')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("Starting ensemble translation process.")
    ensemble = EnsembleTranslator(device=args.device)
    
    if not ensemble.models:
        logger.critical("No models were loaded. Aborting process. Check logs for errors.")
        sys.exit(1)

    vi_translations = []
    if os.path.exists(args.en_csv):
        logger.info(f"Translating English to Vietnamese from {args.en_csv}.")
        vi_translations = ensemble.translate_file(args.en_csv, direction='en2vi')
    else:
        logger.warning(f"English input file not found at {args.en_csv}, skipping EN->VI task.")

    en_translations = []
    if os.path.exists(args.vi_csv):
        logger.info(f"Translating Vietnamese to English from {args.vi_csv}.")
        en_translations = ensemble.translate_file(args.vi_csv, direction='vi2en')
    else:
        logger.warning(f"Vietnamese input file not found at {args.vi_csv}, skipping VI->EN task.")
    
    logger.info("Saving results to output file.")
    max_len = max(len(en_translations), len(vi_translations))
    
    en_translations.extend([""] * (max_len - len(en_translations)))
    vi_translations.extend([""] * (max_len - len(vi_translations)))
    
    results_df = pd.DataFrame({
        'English_Translation': en_translations,
        'Vietnamese_Translation': vi_translations
    })
    
    results_df.to_csv(args.output, index=False, encoding='utf-8')
    logger.info(f"Results saved to {args.output}.")
    
    zip_filename = 'ensemble_results.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(args.output, arcname=os.path.basename(args.output))
    logger.info(f"Output also compressed to {zip_filename}.")
    
    logger.info("Ensemble translation process finished successfully.")

if __name__ == "__main__":
    main()