import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from collections import Counter


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_freq_pre_tokenization_per_chunk(
    chunk: str,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    parts = [p for p in re.split(re.escape("|".join(special_tokens)), chunk) if p != ""]
    freq_pre_tokenization_per_chuck = dict()
    for p in parts:
        for pre_token in re.finditer(PAT, p):
            pre_token_byte = pre_token.group().encode("utf-8")
            key = tuple(pre_token_byte[i:i+1] for i in range(len(pre_token_byte)))
            if key not in freq_pre_tokenization_per_chuck:
                freq_pre_tokenization_per_chuck[key] = 0
            freq_pre_tokenization_per_chuck[key] += 1
    return freq_pre_tokenization_per_chuck


def get_freq_pre_tokenization(
    input_path: str,
    num_processes: int,
    special_tokens: list[str],
    split_special_token: bytes = b"<|endoftext|>",
) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
        with Pool(processes=num_processes) as pool:
            freq_pre_tokenization_dicts = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                freq_pre_tokenization_dicts.append(
                    pool.apply_async(
                        get_freq_pre_tokenization_per_chunk, (chunk, special_tokens)
                    )
                )
            freq_pre_tokenization = Counter(freq_pre_tokenization_dicts[0].get())
            for i in range(1, len(freq_pre_tokenization_dicts)):
                freq_pre_tokenization += Counter(freq_pre_tokenization_dicts[i].get())
            freq_pre_tokenization = dict(freq_pre_tokenization)
    return freq_pre_tokenization


if __name__ == '__main__':
    ## Usage
    import time
    start_time = time.time()
    freq_pre_tokenization_dicts = []
    with open("./tests/fixtures/tinystories_sample_5M.txt", "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        special_tokens = ["<|endoftext|>"]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            parts = [p for p in re.split(re.escape("|".join(special_tokens)), chunk) if p != ""]
            freq_pre_tokenization_per_chuck = dict()
            for p in parts:
                for pre_token in re.finditer(PAT, p):
                    pre_token_byte = pre_token.group().encode("utf-8")
                    key = tuple(pre_token_byte[i:i+1] for i in range(len(pre_token_byte)))
                    if key not in freq_pre_tokenization_per_chuck:
                        freq_pre_tokenization_per_chuck[key] = 0
                    freq_pre_tokenization_per_chuck[key] += 1
            freq_pre_tokenization_dicts.append(freq_pre_tokenization_per_chuck)
        freq_pre_tokenization = Counter(freq_pre_tokenization_dicts[0])
        for i in range(1, len(freq_pre_tokenization_dicts)):
            freq_pre_tokenization += Counter(freq_pre_tokenization_dicts[i])
        freq_pre_tokenization = dict(freq_pre_tokenization)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"No multiprocessing - Execution time: {elapsed_time} seconds")


    start_time = time.time()
    freq_pre_tokenization1 = get_freq_pre_tokenization("./tests/fixtures/tinystories_sample_5M.txt", 4, ["<|endoftext|>"])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"With multiprocessing - Execution time: {elapsed_time} seconds")

    assert freq_pre_tokenization == freq_pre_tokenization1, "FAILED - 2 runs have different results!"