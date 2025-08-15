import unicodedata
import sentencepiece as spm
from underthesea import word_tokenize
from tqdm import tqdm
from typing import List

class MedicalNMTPreprocessor:
    def __init__(self, vocab_size=16000, model_prefix='medical_nmt'):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp_model = None

        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
            '<2en>': 4,
            '<2vi>': 5
        }

    def preprocess_vietnamese(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text.strip())

        try:
            text = word_tokenize(text, format="text")
        except:
            pass

        words = text.split()
        processed_words = []
        for word in words:
            if word.isupper() and len(word) > 1:
                processed_words.append(word)
            else:
                processed_words.append(word.lower())

        return ' '.join(processed_words)

    def preprocess_english(self, text: str) -> str:
        text = text.strip()

        words = text.split()
        processed_words = []
        for word in words:
            if word.isupper() and len(word) > 1:
                processed_words.append(word)
            else:
                processed_words.append(word.lower())

        return ' '.join(processed_words)

    def prepare_training_data(self, en_file: str, vi_file: str):
        print("Preprocessing training data...")

        vi_lines = []
        with open(vi_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing VI"):
                processed = self.preprocess_vietnamese(line)
                if processed:
                    vi_lines.append(processed)

        with open('preprocessed_train.vi.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(vi_lines))

        en_lines = []
        with open(en_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing EN"):
                processed = self.preprocess_english(line)
                if processed:
                    en_lines.append(processed)

        with open('preprocessed_train.en.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(en_lines))

        print(f"Preprocessed {len(en_lines)} EN and {len(vi_lines)} VI sentences")

    def train_sentencepiece(self, en_file: str, vi_file: str):
        self.prepare_training_data(en_file, vi_file)

        print(f"Training SentencePiece model with vocab size {self.vocab_size}...")

        spm.SentencePieceTrainer.train(
            input=['preprocessed_train.en.txt', 'preprocessed_train.vi.txt'],
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=['<2en>', '<2vi>'],
            shuffle_input_sentence=True,
            max_sentence_length=512,
            num_threads=8
        )

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f'{self.model_prefix}.model')
        print(f"SentencePiece model trained. Vocab size: {self.sp_model.vocab_size()}")

    def encode(self, text: str, lang: str = None, add_special: bool = True) -> List[int]:
        if not self.sp_model:
            raise ValueError("SentencePiece model not trained/loaded")

        tokens = self.sp_model.encode(text)

        if add_special:
            if lang == 'en':
                tokens = [self.special_tokens['<2en>']] + tokens
            elif lang == 'vi':
                tokens = [self.special_tokens['<2vi>']] + tokens

            tokens = [self.special_tokens['<s>']] + tokens + [self.special_tokens['</s>']]

        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        if skip_special:
            tokens = [t for t in tokens if t not in self.special_tokens.values()]

        return self.sp_model.decode(tokens)

    def batch_encode(self, texts: List[str], lang: str = None) -> List[List[int]]:
        return [self.encode(text, lang) for text in texts]

    def batch_decode(self, token_lists: List[List[int]], skip_special: bool = True) -> List[str]:
        return [self.decode(tokens, skip_special) for tokens in token_lists]