import torch
from tqdm import tqdm
from custom_tokenizer import CharacterTokenizer

class DataLoader:
    def __init__(self, train_input_ids, valid_input_ids, test_input_ids, block_size, tokenizer: CharacterTokenizer):
        self.train_input_ids = train_input_ids
        self.valid_input_ids = valid_input_ids
        self.test_input_ids = test_input_ids
        self.block_size = block_size
        self.tokenizer = tokenizer

        # Precompute valid starting indices for each dataset
        self.valid_starts = {
            'train': self._get_valid_starts(self.train_input_ids),
            'valid': self._get_valid_starts(self.valid_input_ids),
            'test': self._get_valid_starts(self.test_input_ids),
        }

    def _get_valid_starts(self, data):
        # Identify indices that correspond to a space or newline
        valid_indices = [
            i+1 for i in range(len(data) - self.block_size)
            if data[i].item() in {self.tokenizer.token_to_idx[' '], self.tokenizer.token_to_idx['\n']}
        ]
        return valid_indices

    def get_batch(self, split: str, batch_size: int):
        # Use precomputed valid starting indices
        valid_indices = self.valid_starts[split]
        ix = torch.tensor(valid_indices)[torch.randint(len(valid_indices), (batch_size,))]
        x = torch.stack([
            torch.cat((
                torch.tensor([self.tokenizer.token_to_idx[self.tokenizer.start_token]]),
                self.train_input_ids[i:i + self.block_size-1]
            )) for i in ix
        ])
        y = torch.stack([self.train_input_ids[i: i + self.block_size] for i in ix])
        return x, y

    def iterator(self, split: str, batch_size: int):
        # Use precomputed valid starting indices
        valid_indices = self.valid_starts[split]
        for j in tqdm(range(0, len(valid_indices), batch_size)):
            batch_indices = valid_indices[j:j + batch_size]
            x = torch.stack([self.train_input_ids[i:i + self.block_size] for i in batch_indices])
            y = torch.stack([self.train_input_ids[i + 1:i + self.block_size + 1] for i in batch_indices])
            yield x, y
