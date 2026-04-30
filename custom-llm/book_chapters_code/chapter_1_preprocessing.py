import urllib.request
import os
import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader

INSPECT = os.environ.get("INSPECT") in ["yes","true","1"]
EOT_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"
SPLIT_REGEX = r'([,.:;?_!"()\']|--|\s)'

def download_sample(url: str, file_path: str):
    if not os.path.isfile(file_path):
        urllib.request.urlretrieve(url, file_path)
    else:
        print(f"{file_path} already exists, skipping download")
    if INSPECT:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        print(f"Total number of characters: {len(raw_text)}")
        print(f"First 99 characters: ", raw_text[:99])
    return file_path

def tokenize_regexp(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        tokens = map(lambda r: r.strip(), re.split(SPLIT_REGEX, text))
        tokens = list(filter(lambda s: len(s) > 0, tokens))
        if INSPECT:
            print(f"First 99 tokens: {tokens[:99]}")
    return tokens

def create_vocab(tokens: list[str]):
    all_words = sorted(set(tokens))
    vocab_size = len(all_words)
    print(f"Vocabulary size: {vocab_size}")
    all_words.extend([EOT_TOKEN, UNK_TOKEN])
    vocab = {token:idx for idx,token in enumerate(all_words)}
    if INSPECT:
        print(f"First 50 items in vocabulary")
        print([item for _,item in enumerate(vocab.items())][:50])
    return vocab

class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str,int]):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text: str):
        preprocessed = map(lambda r: r.strip(), re.split(SPLIT_REGEX, text))
        preprocessed = list(filter(lambda s: len(s) > 0, preprocessed))
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids: list[int]):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # removes extra space before punctuation
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str,int]):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text: str):
        preprocessed = map(lambda r: r.strip(), re.split(SPLIT_REGEX, text))
        preprocessed = list(filter(lambda s: len(s) > 0, preprocessed))
        preprocessed = [item if item in self.str_to_int else UNK_TOKEN for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids: list[int]):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # removes extra space before punctuation
        return text

class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, input_window: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={EOT_TOKEN})
        
        for i in range(0, len(token_ids) - input_window, stride):
            input_chunk = token_ids[i:i + input_window]
            target_chunk = token_ids[i + 1:i + input_window + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> tuple[int,int]:
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt: str, tokenizer_name: str, batch_size: int = 4, input_window: int = 256, stride: int = 128, shuffle: bool = True, drop_last: bool = True, num_workers: int = 0):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    dataset = GPTDatasetV1(txt, tokenizer, input_window, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader
        

if __name__ == "__main__":
    samples = [
        {
        'url': "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
        'filename': "the-verdict.txt"
        }
    ]
    samples_dir = os.path.join("data", "samples")
    if not os.path.isdir(samples_dir):
        os.mkdir("data/samples")

    print(f"tiktoken version: {version("tiktoken")}")
    print(f"supported tokenizers: {tiktoken.list_encoding_names()}")
    print(f"torch version: {version("torch")}")
    print(f"processing {len(samples)} samples")
    
    for s in samples:
        sample_url = s['url']
        print(f"="*(len(sample_url) + 11))
        print(f"Processing {s['url']}")
        print(f"="*(len(sample_url) + 11))
        fp = download_sample(sample_url, os.path.join(samples_dir, s['filename']))
        s['path'] = fp
        tokens = tokenize_regexp(fp)
        vocab = create_vocab(tokens)

        simple_tokenizer = SimpleTokenizerV1(vocab)
        if INSPECT:
            with open(fp, "r", encoding="utf-8") as f:
                try:
                    encoded = simple_tokenizer.encode(f.read(99))
                    print(f"Simple tokenizer v1 output of first 99 characters: {encoded}")
                    decoded = simple_tokenizer.decode(encoded)
                    print(f"Decoded output from simple tokenizer v1: {decoded}")
                except KeyError as e:
                    print(f"Unknown token in vocabulary: {e}")
        simple_tokenizer_v2 = SimpleTokenizerV2(vocab)
        if INSPECT:
            with open(fp, "r", encoding="utf-8") as f:
                encoded = simple_tokenizer_v2.encode(f.read(99))
                print(f"Simple tokenizer v2 output of first 99 characters: {encoded}")
                decoded = simple_tokenizer_v2.decode(encoded)
                print(f"Decoded output from simple tokenizer v2: {decoded}")

        bpe_tokenizer = tiktoken.get_encoding("gpt2")
        with open(fp, "r", encoding="utf-8") as f:  
            tokens = bpe_tokenizer.encode(f.read(), allowed_special={EOT_TOKEN})
            if INSPECT:
                print(f"First 99 BPE tokens: {tokens[:99]}")
                print(f"Decoded first 99 tokens: {bpe_tokenizer.decode(tokens[:99])}")
                print(f"Length of encoded text: {len(tokens)}")

        if INSPECT:
            context_size = 4
            x = tokens[:context_size]
            y = tokens[1:context_size + 1]
            print(f"x: {x} / {bpe_tokenizer.decode(x)}")
            print(f"y: {y} / {bpe_tokenizer.decode(y)}")
            enc_sample = tokens[:50]

            for i in range(1, context_size+1):
                context = enc_sample[:i]
                desired = enc_sample[i]
                print(f"{context} / {bpe_tokenizer.decode(context)}", "---->", f"{desired} / {bpe_tokenizer.decode([desired])}")

        if INSPECT:
            with open(fp, "r", encoding="utf-8") as f:
                bpe_dataloader = create_dataloader_v1(f.read(), tokenizer_name="gpt2", input_window=4, stride=4, batch_size=8, shuffle=False)
                data_iter = iter(bpe_dataloader)
                first_batch = next(data_iter)
                print("First batch:\n", first_batch)
                inputs, targets = next(data_iter)
                print("Inputs:\n", inputs)
                print("\nTargets:\n", targets)

        if INSPECT:
            vocab_size = 6
            output_dim = 3
            torch.manual_seed(123)
            embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
            print(f"Embedding layer:\n", embedding_layer.weight)
            print(f"Embedding layer applied: {embedding_layer(torch.tensor([3]))}")

        # actual embedding
        with open(fp, "r", encoding="utf-8") as f:
            vocab_size = 50257
            output_dim = 256
            context_length = 4
            pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
            token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
            dataloader = create_dataloader_v1(f.read(), tokenizer_name="gpt2", input_window=4, stride=4, batch_size=8, shuffle=False)
            for inputs,targets in dataloader:
                token_embeddings: torch.Tensor = token_embedding_layer(inputs)
                pos_embeddings: torch.Tensor = pos_embedding_layer(torch.arange(context_length))
                output = token_embeddings + pos_embeddings
                print("Output dimensions:", output.shape)
                break # TODO: how to combine these?
