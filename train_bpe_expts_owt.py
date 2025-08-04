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
    input_path = "./downloads/data/owt_train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
    )

    # save merges
    with open("train-bpe-owt-merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(gpt2_byte_encode(merge[0]) + ' ' + gpt2_byte_encode(merge[1]) + '\n')

    # save vocab
    vocab_to_save = {gpt2_byte_encode(v): k for k, v in vocab.items()}
    with open("train-bpe-owt-vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, indent=4, sort_keys=True, ensure_ascii=False)


# /usr/bin/time -l uv run train_bpe_expts_owt.py                                                                                                                                                            1 ✘ │ base  │ at 22:35:58 
    #  8027.54 real      9776.08 user       127.78 sys
    #      10387734528  maximum resident set size
    #                0  average shared memory size
    #                0  average unshared data size
    #                0  average unshared stack size
    #         26505253  page reclaims
    #             4933  page faults
    #                0  swaps
    #                0  block input operations
    #                0  block output operations
    #                1  messages sent
    #                1  messages received
    #                1  signals received
    #           906385  voluntary context switches
    #          3462555  involuntary context switches
    #        157167906  instructions retired
    #         85398807  cycles elapsed
    #         10404672  peak memory footprint