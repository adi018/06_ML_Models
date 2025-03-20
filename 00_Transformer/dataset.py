import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        """
        Params:
        - ds: A list of dictionaries. Each dictionary should have a 'translation' key that contains the translation of the text in different languages.
        - tokenizer_src: A SentencePiece tokenizer for the source language
        - tokenizer_tgt: A SentencePiece tokenizer for the target language
        - src_lang: The source language
        - tgt_lang: The target language
        - seq_len: The maximum length of the sequence
        Returns:
        - A dictionary with the following
            - encoder_input: The input to the encoder (seq_len)
            - decoder_input: The input to the decoder (seq_len)
            - encoder_mask: The mask for the encoder input (1, 1, seq_len)
            - decoder_mask: The mask for the decoder input (1, seq_len) & (1, seq_len, seq_len)
            - label: The label for the decoder (seq_len)
            - src_text: The source text
            - tgt_text: The target text 
        """
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # can use either tokenizer_src or tokenizer_tgt to get the sos, eos, and pad tokens
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s> tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # We will add <s> token as the first token but not </s> because we want to predict it

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        # Add SOS and EOS tokens to the encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        # Add SOS token to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        # Add EOS token to the label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        """
        The encoder_mask ensures that the padding tokens in the encoder input are ignored during the self-attention computation. Padding tokens are added to make all sequences in a batch the same length, but they do not carry meaningful information and should not influence the attention mechanism.

        The decoder_mask serves two purposes:

        - Ignore Padding Tokens: Like the encoder_mask, it ensures that padding tokens in the decoder input are ignored during self-attention.
        - Causal Masking: It enforces the autoregressive property of the decoder by ensuring that each position in the sequence can only attend to itself and the positions before it (not future positions).

        """

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    """
    Create a mask to prevent the decoder from looking ahead.
    Params:
    - size: The size of the mask
    Returns:
    - A tensor with a triangular matrix of ones and zeros"""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    # Every where where mask is 1, the value will be masked
    # Every where where mask is 0, the value will be kept as is mask == 0 is True
    # We reject upper triangle by making it masked i.e. fill it with 0 and
    # we return the diagonal and the lower triangle with the true values
    return mask == 0