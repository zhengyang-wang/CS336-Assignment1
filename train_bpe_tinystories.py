import json
from tests.adapters import run_train_bpe
from tests.common import gpt2_bytes_to_unicode


GPT2_BYTE_ENCODER = gpt2_bytes_to_unicode()


def gpt2_byte_encode(input_bytes: bytes) -> str:
    result = ""
    for byte in input_bytes:
        result += GPT2_BYTE_ENCODER[byte]
    return result


if __name__ == '__main__':
    input_path = "./downloads/data/TinyStoriesV2-GPT4-train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    # save merges
    with open("train-bpe-tinystories-merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(gpt2_byte_encode(merge[0]) + ' ' + gpt2_byte_encode(merge[1]) + '\n')

    # save vocab
    vocab_to_save = {gpt2_byte_encode(v): k for k, v in vocab.items()}
    with open("train-bpe-tinystories-vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, indent=4, sort_keys=True, ensure_ascii=False)


# /usr/bin/time -l uv run train_bpe_tinystories.py                                                                                                                                                 ✔ │ took 31s │ base  │ at 22:28:26 
#        83.43 real       448.61 user        14.61 sys
#           3488923648  maximum resident set size
#                    0  average shared memory size
#                    0  average unshared data size
#                    0  average unshared stack size
#              2349594  page reclaims
#                   12  page faults
#                    0  swaps
#                    0  block input operations
#                    0  block output operations
#                    1  messages sent
#                    1  messages received
#                    1  signals received
#                35832  voluntary context switches
#               238359  involuntary context switches
#             89712312  instructions retired
#             37534755  cycles elapsed
#             10502976  peak memory footprint