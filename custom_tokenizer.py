import pickle
from abc import ABC, abstractmethod
from typing import List, Dict

import torch


class TokenizerBase(ABC):
    """
    Abstract base class for text tokenizers that convert text into numerical tokens and vice versa.

    This class provides the basic structure for implementing various tokenization strategies.
    It includes methods for encoding text to token IDs, decoding token IDs back to text,
    and handling special tokens like start, end, padding, and unknown tokens.

    Attributes:
        special_tokens (List[str]): List of special tokens used by the tokenizer
        idx_to_token (Dict[int, str]): Mapping from token indices to token strings
        token_to_idx (Dict[str, int]): Mapping from token strings to token indices
    """

    def __init__(self) -> None:
        self.special_tokens: List[str] = []
        self.idx_to_token: Dict[int, str] = {}
        self.token_to_idx: Dict[str, int] = {}

    @property
    @abstractmethod
    def start_token(self) -> str:
        """Token representing the beginning of the sequence."""
        pass

    @property
    @abstractmethod
    def end_token(self) -> str:
        """Token representing the end of the sequence."""
        pass

    @property
    @abstractmethod
    def pad_token(self) -> str:
        """Token used for padding sequences to equal length."""
        pass

    @property
    @abstractmethod
    def unk_token(self) -> str:
        """Token used for unknown or out-of-vocabulary characters."""
        pass

    @property
    def num_tokens(self) -> int:
        """
        Get the total number of unique tokens in the tokenizer's vocabulary.

        Returns:
            int: The total number of tokens in the vocabulary, including special tokens.
        """
        return len(self.idx_to_token)

    @abstractmethod
    def train(self, file_path: str, n_tokens: int = -1, start_token: str = "<s>",
              end_token: str = "</s>", pad_token: str = "<p>", unk_token: str = "<unk>") -> None:
        """
        Train the tokenizer on a dataset to build its vocabulary.

        Args:
            file_path: Path to the training data file
            n_tokens: Maximum number of tokens to include in the vocabulary (-1 for no limit)
            start_token: Token to represent the start of a sequence
            end_token: Token to represent the end of a sequence
            pad_token: Token to use for padding sequences to equal length
            unk_token: Token to use for unknown or out-of-vocabulary characters

        Note:
            The implementation should populate idx_to_token and token_to_idx mappings
            and set up special tokens based on the training data.
        """
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            text: The input text to encode
            add_special_tokens: Whether to add start and end tokens to the encoded sequence

        Returns:
            A list of integer token IDs representing the encoded text
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            tokens: List of integer token IDs to decode

        Returns:
            The decoded text string
        """
        pass

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of text strings into token IDs.

        Args:
            texts: List of input text strings to encode

        Returns:
            List of lists containing token IDs for each input text
        """
        pass

    def decode_batch(self, batch_tokens: List[List[int]]) -> List[str]:
        """
        Decode a batch of token ID sequences back into text strings.

        Args:
            batch_tokens: List of lists containing token IDs to decode

        Returns:
            List of decoded text strings
        """
        pass

    def save(self, filepath: str) -> None:
        """
        Save the tokenizer instance to a file using pickle serialization.

        Args:
            filepath: Path where the serialized tokenizer should be saved

        Note:
            This method uses pickle for serialization, which may not be secure for
            untrusted data sources.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'TokenizerBase':
        """
        Load a tokenizer instance from a file using pickle deserialization.

        Args:
            filepath: Path to the saved tokenizer file

        Returns:
            The loaded tokenizer instance

        Note:
            This method uses pickle for deserialization, which may not be secure for
            untrusted data sources.
        """
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filepath}")
        return model

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens without converting to token IDs.

        Args:
            text: Input text to tokenize

        Returns:
            List of token strings
        """
        return [self.decode(self.encode(i, add_special_tokens=False)) for i in text.split()]


class CharacterTokenizer(TokenizerBase):
    """
    A character-level tokenizer that treats each character as a separate token.

    This tokenizer creates a vocabulary from individual characters in the training data
    and handles text encoding/decoding at the character level. It includes special tokens
    for sequence start, end, padding, and unknown characters.
    """

    def __init__(self) -> None:
        """Initialize the character tokenizer with default special tokens."""
        super().__init__()
        self._start_token: str = "<s>"
        self._end_token: str = "</s>"
        self._pad_token: str = "<p>"
        self._unk_token: str = "<unk>"

    @property
    def start_token(self) -> str:
        return self._start_token

    @property
    def end_token(self) -> str:
        return self._end_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def unk_token(self) -> str:
        return self._unk_token

    def train(self, file_path: str, n_tokens: int = -1, start_token: str = "<s>",
              end_token: str = "</s>", pad_token: str = "<p>", unk_token: str = "<unk>") -> None:
        """
        Train the character tokenizer on a text file.

        Args:
            file_path: Path to the training data file
            n_tokens: Maximum number of tokens to include (not used in character tokenizer)
            start_token: Token to represent sequence start
            end_token: Token to represent sequence end
            pad_token: Token to use for padding
            unk_token: Token to use for unknown characters

        Note:
            The character tokenizer creates a vocabulary from all unique characters
            in the training data plus the special tokens.
        """
        with open(file_path, 'r') as f:
            data = ''.join(f.readlines())
        self.special_tokens = [start_token, end_token, pad_token, unk_token]
        self._start_token = start_token
        self._end_token = end_token
        self._pad_token = pad_token
        self._unk_token = unk_token

        tokens = self.special_tokens + list(set(data))
        self.idx_to_token = {idx: token for idx, token in enumerate(tokens)}
        self.token_to_idx = {token: idx for idx, token in enumerate(tokens)}

    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        Encode text into character-level token IDs.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add start and end tokens

        Returns:
            List of integer token IDs
        """
        token_ids = [self.token_to_idx.get(i, self.token_to_idx[self.unk_token]) for i in text]
        if add_special_tokens:
            token_ids.insert(0, self.token_to_idx[self.start_token])
            token_ids.append(self.token_to_idx[self.end_token])
        return torch.tensor(token_ids)

    def pad(self, token_ids: List[List[int]]) -> List[List[int]]:
        """
        Pad a batch of token ID sequences to the same length.

        Args:
            token_ids: List of token ID sequences to pad

        Returns:
            Padded token ID sequences of equal length
        """
        max_length = max(len(sublist) for sublist in token_ids)
        pad_token_id = self.token_to_idx[self.pad_token]
        return [batch + [pad_token_id] * (max_length - len(batch)) for batch in token_ids]

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts into padded token ID sequences.

        Args:
            texts: List of input texts to encode

        Returns:
            List of padded token ID sequences
        """
        return torch.tensor(self.pad([self.encode(text).tolist() for text in texts]))

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode a sequence of token IDs back into text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        return ''.join([self.idx_to_token[i.item()] for i in token_ids])

    def decode_batch(self, batch_tokens: torch.Tensor) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            batch_tokens: List of token ID sequences to decode

        Returns:
            List of decoded text strings
        """
        return [self.decode(i) for i in batch_tokens]
