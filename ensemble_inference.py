import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
import zipfile
import logging
from collections import Counter

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from sacrebleu import sentence_bleu
except ImportError:
    os.system("pip install transformers sacrebleu sentencepiece")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from sacrebleu import sentence_bleu

class EnsembleTranslator:
    def __init__(self, models_config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.weights = {}
        if models_config is None:
            models_config = {
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
                    'model_name': 'VietAI/envit5-translation',
                    'type': 'envit5'
                }
            }
        self.models_config = models_config
        self.setup_logging()
        self.load_models()
        
    def setup_logging(self):
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ensemble_inference.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        self.logger.info("="*60)
        self.logger.info("Loading Ensemble Models")
        self.logger.info("="*60)
        for model_id, config in self.models_config.items():
            self.logger.info(f"Loading {model_id} (weight: {config['weight']})...")
            try:
                if config['type'] in ['nllb', 'envit5']:
                    if '3.3B' in config['model_name'] and self.device.type == 'cuda':
                        if torch.cuda.get_device_properties(0).total_memory < 10e9:
                            self.logger.warning(f"Skipping {model_id} - insufficient GPU memory.")
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
                self.logger.info(f"✓ Loaded {model_id}")
            except Exception as e:
                self.logger.error(f"Failed to load {model_id}: {e}")
        total_weight = sum(m['weight'] for m in self.models.values())
        if total_weight > 0:
            for model_id in self.models:
                self.models[model_id]['weight'] /= total_weight
        self.logger.info(f"Ensemble ready with {len(self.models)} models.")
        self.logger.info("Normalized weights:")
        for model_id, model_info in self.models.items():
            self.logger.info(f"  {model_id}: {model_info['weight']:.3f}")
    
    def translate_single_model(self, text: str, model_id: str, direction='en2vi') -> str:
        model_info = self.models[model_id]
        try:
            with torch.no_grad():
                tokenizer = model_info['tokenizer']
                model = model_info['model']
                if model_info['type'] == 'nllb':
                    if direction == 'en2vi':
                        tokenizer.src_lang = "eng_Latn"
                        tgt_lang = "vie_Latn"
                    else:
                        tokenizer.src_lang = "vie_Latn"
                        tgt_lang = "eng_Latn"
                    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                    generated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )
                    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                elif model_info['type'] == 'envit5':
                    if direction == 'en2vi':
                        prompt = f"en: {text}"
                    else:
                        prompt = f"vi: {text}"
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                    generated_tokens = model.generate(**inputs, max_length=512, num_beams=4)
                    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Error during translation with {model_id}: {e}")
            return ""
    
    def ensemble_translate(self, text: str, direction: str) -> str:
        if not text or not text.strip():
            return ""
        if direction == 'en2vi':
            if 'nllb-3.3b' in self.models:
                return self.translate_single_model(text, 'nllb-3.3b', direction)
            elif 'nllb-1.3b' in self.models:
                return self.translate_single_model(text, 'nllb-1.3b', direction)
            else:
                self.logger.error("No NLLB model available for EN→VI.")
                return ""
        elif direction == 'vi2en':
            if 'vinai-envit5' in self.models:
                return self.translate_single_model(text, 'vinai-envit5', direction)
            else:
                self.logger.error("No ViT5 model available for VI→EN.")
                return ""
        else:
            self.logger.error(f"Unknown direction: {direction}")
            return ""
    
    def translate_file(self, input_file: str, direction='en2vi') -> List[str]:
        self.logger.info(f"Starting file translation for {input_file} ({direction})")
        try:
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
                texts = df[df.columns[0]].fillna("").astype(str).tolist()
            else:
                with open(input_file, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f]
        except FileNotFoundError:
            self.logger.error(f"Input file not found: {input_file}")
            return []
        translations = []
        for text in tqdm(texts, desc=f"Ensemble Translating ({direction})"):
            translation = self.ensemble_translate(text, direction)
            translations.append(translation)
        return translations

def main():
    print("="*70)
    print("Ensemble Translation with NLLB-3.3B, NLLB-1.3B, and vinai-enviT5")
    print("="*70)
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble translation script.")
    parser.add_argument('--en-csv', default='en.csv', help='Path to the English source CSV file.')
    parser.add_argument('--vi-csv', default='vi.csv', help='Path to the Vietnamese source CSV file.')
    parser.add_argument('--output', default='ensemble_results.csv', help='Name of the output CSV file.')
    args = parser.parse_args()
    print("\nInitializing the ensemble of translators...")
    ensemble = EnsembleTranslator()
    if not ensemble.models:
        print("\nNo models were loaded. This could be due to insufficient memory or other errors.")
        print("Please check the logs in the 'logs/' directory.")
        sys.exit(1)
    vi_translations = []
    if os.path.exists(args.en_csv):
        print(f"\nTranslating English to Vietnamese from: {args.en_csv}")
        vi_translations = ensemble.translate_file(args.en_csv, direction='en2vi')
    else:
        print(f"\nWarning: English input file not found at {args.en_csv}. Skipping EN->VI translation.")
    en_translations = []
    if os.path.exists(args.vi_csv):
        print(f"\nTranslating Vietnamese to English from: {args.vi_csv}")
        en_translations = ensemble.translate_file(args.vi_csv, direction='vi2en')
    else:
        print(f"\nWarning: Vietnamese input file not found at {args.vi_csv}. Skipping VI->EN translation.")
    print("\nProcessing and saving results...")
    max_len = max(len(en_translations), len(vi_translations))
    en_translations.extend([""] * (max_len - len(en_translations)))
    vi_translations.extend([""] * (max_len - len(vi_translations)))
    results_df = pd.DataFrame({
        'English_Translation_from_VI': en_translations,
        'Vietnamese_Translation_from_EN': vi_translations
    })
    results_df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"✓ Results successfully saved to {args.output}")
    zip_filename = 'ensemble_results.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(args.output, arcname='results.csv')
    print(f"✓ Results also compressed into {zip_filename}")
    print("\n" + "="*70)
    print("Ensemble Translation Process Complete!")
    print(f"Submit the '{zip_filename}' file.")
    print("="*70)

if __name__ == "__main__":
    main()
