import os
import json
import regex as re
from collections.abc import Iterable, Iterator
from tests.common import gpt2_bytes_to_unicode
from cs336_basics.pretokenization_example import find_chunk_boundaries


class Tokenizer(object):
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # pre-tokenization regex pattern
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # add special tokens to vocab if not existed
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        self.vocab_int_to_bytes = vocab
        self.vocab_bytes_to_int = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        # need gpt2_byte_decoder here,
        # as I use gpt2_bytes_to_unicode before saving vocab and merges to files
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }

        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
    

    def encode(self, text: str) -> list[int]:
        # handle special tokens
        if self.special_tokens:
            tok_pat = '(' + '|'.join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True))) + ')'
            parts = [p for p in re.split(tok_pat, text) if p != '']
        else:
            parts = [text]
        
        output = list()
        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                output += [self.vocab_bytes_to_int[part.encode("utf-8")]]
            else:
                for pre_tokenized_token in self.pre_tokenize(part): # step 1: pre-tokenize
                    pre_tokenized_token_bytes_tuple = self.apply_merges_on_pre_tokenized_token(pre_tokenized_token)
                    # step 3: map to int
                    pre_tokenized_token_encoded = [self.vocab_bytes_to_int[k] for k in pre_tokenized_token_bytes_tuple]
                    output += pre_tokenized_token_encoded
        return output


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            current_encoded = self.encode(text)
            for _id in current_encoded:
                yield _id
    

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab_int_to_bytes[k] for k in ids]).decode('utf-8', errors='replace')
    

    def pre_tokenize(self, text: str) -> Iterator[str]:
        return re.finditer(self.PAT, text)
    
    
    def apply_merges_on_pre_tokenized_token(self, pre_tokenized_token: str) -> tuple[bytes]:
        pre_tokenized_token_bytes = pre_tokenized_token.group().encode("utf-8")
        pre_tokenized_token_bytes_tuple = tuple([
            pre_tokenized_token_bytes[i:i+1] for i in range(len(pre_tokenized_token_bytes))])
        # step 2: apply the merges
        ## find the first matched merge and apply, until no match can be found
        while True:
            found = False
            for merge in self.merges:
                if merge[0] in pre_tokenized_token_bytes_tuple:
                    for i, b in enumerate(pre_tokenized_token_bytes_tuple):
                        if b == merge[0] and \
                            i < len(pre_tokenized_token_bytes_tuple)-1 and \
                                pre_tokenized_token_bytes_tuple[i+1] == merge[1]:
                            found = True
                            pre_tokenized_token_bytes_tuple = \
                                pre_tokenized_token_bytes_tuple[:i] \
                                + tuple([b"".join(merge)]) \
                                + pre_tokenized_token_bytes_tuple[i+2:]
                            break
                if found:
                    break
            if not found:
                break
        return pre_tokenized_token_bytes_tuple
    

if __name__ == '__main__':
    from tests.common import FIXTURES_PATH

    # quick test
    # vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    # input_str = 'the cat ate <|endoftext|><|endoftext|> the<|endoftext|>\n\n'
    # tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>", "<|endoftext|><|endoftext|>"])

    # test gpt-2
    VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
    tokenizer = Tokenizer.from_files(vocab_filepath=VOCAB_PATH, merges_filepath=MERGES_PATH, special_tokens=["<|endoftext|>"])

    # input_str = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>\n\n"

    # print(tokenizer.encode(input_str))
    # print(tokenizer.decode(tokenizer.encode(input_str)))
    # tokenized_string = [tokenizer.decode([x]) for x in tokenizer.encode(input_str)]
    # print(tokenized_string)
    # print(tokenizer.decode(tokenizer.encode(input_str)) == input_str)

    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)

    print(tokenizer.decode(all_ids))