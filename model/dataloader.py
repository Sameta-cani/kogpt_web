import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DataCollatorForLanguageModeling
from utils import calculate_normalized_weight_with_smoothing

class GPTDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        # 데이터 불러오기 및 가중치 계산
        data = pd.read_csv(file_path)
        data['Weighted_Score'] = data.apply(lambda row: calculate_normalized_weight_with_smoothing(row['votes'], row['comment_nums'], row['scraps']), axis=1)
        weights = torch.tensor(data['Weighted_Score'].values, dtype=torch.float32)
        weights = (weights - weights.min()) / (weights.max() - weights.min()) + 1
        
        # 토큰화
        concats = [
            label + "|" + text for label, text in zip(data["label"], data["content"])
        ]
        self.item = tokenizer(
            concats,
            return_tensors="pt", # Return PyTorch objects
            padding="max_length",
            truncation=True,
            max_length=64,
        )["input_ids"]
        self.length = len(concats)
        self.sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    def __getitem__(self, i):
        return self.item[i]
    
    def __len__(self):
        return self.length
    
def GPTDataLoader(tokenizer, file_path, batch_size):
    data = GPTDataset(tokenizer, file_path)
    return DataLoader(data, batch_size=batch_size, sampler=data.sampler,
                      collate_fn=DataCollatorForLanguageModeling(
                          tokenizer=tokenizer,
                          mlm=False
                      ))