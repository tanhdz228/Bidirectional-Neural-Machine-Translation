import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random

class MedicalTranslationDataset(Dataset):
    def __init__(self, en_file, vi_file, preprocessor, max_length=256, is_train=True):
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.is_train = is_train
        self.data = []
        
        print(f"Loading {'training' if is_train else 'validation'} data...")
        
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(vi_file, 'r', encoding='utf-8') as f_vi:
            
            en_lines = f_en.readlines()
            vi_lines = f_vi.readlines()
            
            assert len(en_lines) == len(vi_lines), \
                f"Mismatched file lengths: EN={len(en_lines)}, VI={len(vi_lines)}"
            
            for en_line, vi_line in tqdm(zip(en_lines, vi_lines), 
                                        total=len(en_lines),
                                        desc="Processing sentences"):
                en_text = preprocessor.preprocess_english(en_line.strip())
                vi_text = preprocessor.preprocess_vietnamese(vi_line.strip())
                
                if not en_text or not vi_text:
                    continue
                
                if len(en_text.split()) > max_length or len(vi_text.split()) > max_length:
                    continue
                
                self.data.append({
                    'en': en_text,
                    'vi': vi_text
                })
        
        print(f"Loaded {len(self.data)} parallel sentences")
    
    def __len__(self):
        return len(self.data) * 2
    
    def __getitem__(self, idx):
        actual_idx = idx // 2
        direction = 'en2vi' if idx % 2 == 0 else 'vi2en'
        
        if self.is_train and random.random() < 0.1:
            direction = 'vi2en' if direction == 'en2vi' else 'en2vi'
        
        item = self.data[actual_idx]
        
        if direction == 'en2vi':
            src_tokens = self.preprocessor.encode(item['en'], add_special=False)
            tgt_tokens = self.preprocessor.encode(item['vi'], lang='vi', add_special=True)
        else:
            src_tokens = self.preprocessor.encode(item['vi'], add_special=False)
            tgt_tokens = self.preprocessor.encode(item['en'], lang='en', add_special=True)
        
        src_tokens = [self.preprocessor.special_tokens['<s>']] + src_tokens + \
                     [self.preprocessor.special_tokens['</s>']]
        
        src_tokens = src_tokens[:self.max_length]
        tgt_tokens = tgt_tokens[:self.max_length]
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens[:-1], dtype=torch.long),
            'tgt_out': torch.tensor(tgt_tokens[1:], dtype=torch.long),
            'direction': direction
        }

def collate_fn(batch):
    en2vi_batch = []
    vi2en_batch = []
    
    for item in batch:
        if item['direction'] == 'en2vi':
            en2vi_batch.append(item)
        else:
            vi2en_batch.append(item)
    
    processed_batches = []
    
    for direction_batch, direction_name in [(en2vi_batch, 'en2vi'), (vi2en_batch, 'vi2en')]:
        if not direction_batch:
            continue
        
        src = pad_sequence([item['src'] for item in direction_batch], 
                          batch_first=True, padding_value=0)
        tgt = pad_sequence([item['tgt'] for item in direction_batch], 
                          batch_first=True, padding_value=0)
        tgt_out = pad_sequence([item['tgt_out'] for item in direction_batch], 
                              batch_first=True, padding_value=0)
        
        processed_batches.append({
            'src': src,
            'tgt': tgt,
            'tgt_out': tgt_out,
            'direction': direction_name
        })
    
    return processed_batches

class MonolingualDataset(Dataset):
    def __init__(self, file_path, lang, preprocessor, max_length=256):
        self.preprocessor = preprocessor
        self.lang = lang
        self.max_length = max_length
        self.data = []
        
        print(f"Loading monolingual {lang} data from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {lang}"):
                if lang == 'en':
                    text = preprocessor.preprocess_english(line.strip())
                else:
                    text = preprocessor.preprocess_vietnamese(line.strip())
                
                if text and len(text.split()) <= max_length:
                    self.data.append(text)
        
        print(f"Loaded {len(self.data)} {lang} sentences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.preprocessor.encode(text, add_special=True)
        tokens = tokens[:self.max_length]
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'lang': self.lang
        }