import random
import numpy as np
from typing import TextIO, Iterator
from cs336_basics.tokenizer import Tokenizer
import argparse
import timeit, statistics as stats


TinyStories_train_data_path = './downloads/data/TinyStoriesV2-GPT4-train.txt'
TinyStories_dev_data_path = './downloads/data/TinyStoriesV2-GPT4-valid.txt'
TinyStories_vacab_path = './train-bpe-tinystories-vocab.json'
TinyStories_merges_path = './train-bpe-tinystories-merges.txt'
OpenWebText_train_data_path = './downloads/data/owt_train.txt'
OpenWebText_dev_data_path = './downloads/data/owt_valid.txt'
OpenWebText_vacab_path = './train-bpe-owt-vocab.json'
OpenWebText_merges_path = './train-bpe-owt-merges.txt'


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="script.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("a", help="run program a"); s.set_defaults(func=run_a)
    s = sub.add_parser("b", help="run program b"); s.set_defaults(func=run_b)
    s = sub.add_parser("c", help="run program c"); s.set_defaults(func=run_c)
    s = sub.add_parser("d", help="run program d"); s.set_defaults(func=run_d)
    return p


def to_uint16_and_save(ids: list[int], out_path: str) -> None:
    arr = np.asarray(ids, dtype=np.uint32) # avoid silent wrap
    if arr.size and (arr.max() >= 65536 or arr.min() < 0):
        raise ValueError("Token id out of uint16 range; use uint32 instead.")
    np.save(out_path, arr.astype(np.uint16))


def load_uint16(path: str) -> np.ndarray:
    return np.load(path) # dtype preserved


def segments(fp: TextIO, sep: str = "<|endoftext|>", chunk_size: int = 1 << 30) -> Iterator[str]:
    buf = ""
    while True:
        chunk = fp.read(chunk_size)
        if not chunk: break
        parts = chunk.split(sep)
        buf = parts.pop()
        for p in parts:
            if p:
                yield p
    if buf:
        yield buf


def sample_reservoir(path: str, k: int = 10, sep: str = "<|endoftext|>", seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    with open(path) as f:
        for n, seg in enumerate(segments(f, sep=sep), 1):
            if n <= k:
                out.append(seg)           # fill reservoir
            else:
                j = rng.randrange(n)      # random in [0, n-1]
                if j < k:
                    out[j] = seg
    return out


def run_a(args):
    # (a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyS-
    # tories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these
    # sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
    print('Sample and tokenize TinyStories ...')
    tokenizer = Tokenizer.from_files(
        vocab_filepath=TinyStories_vacab_path,
        merges_filepath=TinyStories_merges_path,
        special_tokens=["<|endoftext|>"])
    TinyStories_random_10 = sample_reservoir(TinyStories_train_data_path, k=10, seed=0)
    TinyStories_random_10_tokenized = [tokenizer.encode(d) for d in TinyStories_random_10]
    total_bytes = sum([len(d.encode("utf-8")) for d in TinyStories_random_10])
    total_tokens = sum([len(tokens) for tokens in TinyStories_random_10_tokenized])
    print(f"TinyStories samples total_bytes: {total_bytes}.")
    print(f"TinyStories samples total_tokens after tokenization: {total_tokens}.")
    print(f"TinyStories tokenizer's compression ratio: {total_bytes/total_tokens}.")
    print('Sample and tokenize TinyStories ... done.')

    print('\n\nSample and tokenize OpenWebText ...')
    tokenizer = Tokenizer.from_files(
        vocab_filepath=OpenWebText_vacab_path,
        merges_filepath=OpenWebText_merges_path,
        special_tokens=["<|endoftext|>"])
    OpenWebText_random_10 = sample_reservoir(OpenWebText_train_data_path, k=10, seed=0) # must set seed for (b)
    OpenWebText_random_10_tokenized = [tokenizer.encode(d) for d in OpenWebText_random_10]
    total_bytes = sum([len(d.encode("utf-8")) for d in OpenWebText_random_10])
    total_tokens = sum([len(tokens) for tokens in OpenWebText_random_10_tokenized])
    print(f"OpenWebText samples total_bytes: {total_bytes}.")
    print(f"OpenWebText samples total_tokens after tokenization: {total_tokens}.")
    print(f"OpenWebText tokenizer's compression ratio: {total_bytes/total_tokens}.")
    print('Sample and tokenize OpenWebText ... done.')


def run_b(args):
    # (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Com-
    # pare the compression ratio and/or qualitatively describe what happens.
    print('Sample and tokenize OpenWebText with TinyStories tokenizer ...')
    tokenizer = Tokenizer.from_files(
        vocab_filepath=TinyStories_vacab_path,
        merges_filepath=TinyStories_merges_path,
        special_tokens=["<|endoftext|>"])
    OpenWebText_random_10 = sample_reservoir(OpenWebText_train_data_path, k=10, seed=0)
    OpenWebText_random_10_tokenized = [tokenizer.encode(d) for d in OpenWebText_random_10]
    total_bytes = sum([len(d.encode("utf-8")) for d in OpenWebText_random_10])
    total_tokens = sum([len(tokens) for tokens in OpenWebText_random_10_tokenized])
    print(f"OpenWebText samples total_bytes: {total_bytes}.")
    print(f"OpenWebText samples total_tokens after tokenization with TinyStories tokenizer: {total_tokens}.")
    print(f"The compression ratio: {total_bytes/total_tokens}.")
    print('Sample and tokenize OpenWebText with TinyStories tokenizer ... done.')


def run_c(args):
    # (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
    # tokenize the Pile dataset (825GB of text)?
    TinyStories_random = sample_reservoir(TinyStories_train_data_path, k=10_000, seed=0)
    total_bytes = sum([len(d.encode("utf-8")) for d in TinyStories_random])
    print(f"TinyStories samples total_bytes: {total_bytes}.")

    def tokenize(tokenizer: "Tokenizer", list_of_texts: list[str] = TinyStories_random):
        all_ids = []
        for _id in tokenizer.encode_iterable(list_of_texts):
            all_ids.append(_id)
        return all_ids

    def bench(fn, *args, repeats=10, number=1, **kwargs):
        t = timeit.repeat(lambda: fn(*args, **kwargs), repeat=repeats, number=number)
        return {
            "runs": repeats,
            "calls_per_run": number,
            "best_s": min(t),
            "median_s": stats.median(t),
            "mean_s": stats.mean(t),
            "stdev_s": stats.pstdev(t),
        }
    
    print('Estimate throughput of TinyStories tokenizer ...')
    tokenizer = Tokenizer.from_files(
        vocab_filepath=TinyStories_vacab_path,
        merges_filepath=TinyStories_merges_path,
        special_tokens=["<|endoftext|>"])
    running_time_stats = bench(tokenize, tokenizer)
    bytes_per_second = total_bytes / running_time_stats["mean_s"]
    print(f"Estimated throughput: {bytes_per_second} bytes/s."
          f"It is projected to {825_000_000_000 / bytes_per_second / 3_600} hours for 825GB.")
    print('Estimate throughput of TinyStories tokenizer ... done.')

    print('\n\nEstimate throughput of OpenWebText tokenizer ...')
    # no need to re-sample from OpenWebText
    tokenizer = Tokenizer.from_files(
        vocab_filepath=OpenWebText_vacab_path,
        merges_filepath=OpenWebText_merges_path,
        special_tokens=["<|endoftext|>"])
    running_time_stats = bench(tokenize, tokenizer)
    bytes_per_second = total_bytes / running_time_stats["mean_s"]
    print(f"Estimated throughput: {bytes_per_second} bytes/s."
          f"It is projected to {825_000_000_000 / bytes_per_second / 3_600} hours for 825GB.")
    print('Estimate throughput of OpenWebText tokenizer ... done.')


def run_d(args):
    # (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and devel-
    # opment datasets into a sequence of integer token IDs. We’ll use this later to train our language
    # model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is
    # uint16 an appropriate choice?

    print('Tokenizing TinyStories ...')
    tokenizer = Tokenizer.from_files(
        vocab_filepath=TinyStories_vacab_path,
        merges_filepath=TinyStories_merges_path,
        special_tokens=["<|endoftext|>"])
    
    print('Tokenizing TinyStories train data ...')
    all_ids = []
    with open(TinyStories_train_data_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    to_uint16_and_save(all_ids, 'TinyStories_train_data_tokenized.npy')

    print('Tokenizing TinyStories dev data ...')
    all_ids = []
    with open(TinyStories_dev_data_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    to_uint16_and_save(all_ids, 'TinyStories_dev_data_tokenized.npy')
    print('Tokenizing TinyStories ... done.')

    print('Tokenizing OpenWebText ...')
    tokenizer = Tokenizer.from_files(
        vocab_filepath=OpenWebText_vacab_path,
        merges_filepath=OpenWebText_merges_path,
        special_tokens=["<|endoftext|>"])
    
    print('Tokenizing OpenWebText train data ...')
    all_ids = []
    with open(OpenWebText_train_data_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    to_uint16_and_save(all_ids, 'owt_train_data_tokenized.npy')

    print('Tokenizing OpenWebText dev data ...')
    all_ids = []
    with open(OpenWebText_dev_data_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    to_uint16_and_save(all_ids, 'owt_dev_data_tokenized.npy')
    print('Tokenizing OpenWebText ... done.')


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
