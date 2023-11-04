import argparse
import itertools
import string
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

from protembed.alphabets import Uniprot21
from protembed.fasta import parse_stream
from protembed.models.modules.packed_sequence import PackedTensorSequences
from protembed.models.poet import PoET
from protembed.msa.sampling import MSASampler, NeighborsSampler

ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()
PBAR_POSITION = 1


T = TypeVar("T", np.ndarray, torch.Tensor)


def append_startstop(x: T, alphabet: Uniprot21) -> T:
    x_ndim = x.ndim
    assert x_ndim in {1, 2}
    if x_ndim == 1:
        x = x[None, :]

    if isinstance(x, torch.Tensor):
        empty_func = torch.empty
    else:
        empty_func = np.empty
    x_ = empty_func((x.shape[0], x.shape[1] + 2), dtype=x.dtype)
    x_[:, 0] = alphabet.start_token
    x_[:, -1] = alphabet.stop_token
    x_[:, 1:-1] = x
    if x_ndim == 1:
        x_ = x_.flatten()
    return x_


def get_seqs_from_fastalike(filepath: Path) -> list[bytes]:
    return [s for _, s in parse_stream(open(filepath, "rb"), upper=False)]


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> np.ndarray:
    return np.vstack(
        [
            alphabet.encode(s.translate(None, delete=ASCII_LOWERCASE_BYTES))
            for s in msa_sequences
        ]
    )


def sample_msa_sequences(
    get_sequence_fn: Callable[[int], bytes],
    sample_idxs: Sequence[int],
    max_tokens: int,
    alphabet: Uniprot21,
    shuffle: bool = True,
    shuffle_seed: Optional[int] = None,
    truncate: bool = True,
) -> list[np.ndarray]:
    assert alphabet.start_token != -1
    assert alphabet.stop_token != -1
    if not shuffle:
        assert shuffle_seed is None

    seqs, total_tokens = [], 0
    for idx in sample_idxs:
        next_sequence = get_sequence_fn(idx)
        seqs.append(append_startstop(alphabet.encode(next_sequence), alphabet=alphabet))
        total_tokens += len(seqs[-1])
        if total_tokens > max_tokens:
            break

    # shuffle order and truncate to max tokens
    if shuffle:
        rng = (
            np.random.default_rng(shuffle_seed)
            if shuffle_seed is not None
            else np.random
        )
        final_permutation = rng.permutation(len(seqs))
    else:
        final_permutation = np.arange(len(seqs))
    final_seqs, total_tokens = [], 0
    for seq in [seqs[i] for i in final_permutation]:
        if truncate and (total_tokens + len(seq) > max_tokens):
            seq = seq[: max_tokens - total_tokens]
        total_tokens += len(seq)
        final_seqs.append(seq)
        if total_tokens >= max_tokens:
            break
    return final_seqs


def jit_warmup(embedding_model: PoET, alphabet: Uniprot21):
    x = b"$WAAAGH*$WAAGW*"
    segment_sizes = [8, 7]
    x = alphabet.encode(x)  # encode x into the uniprot21 alphabet
    x = torch.from_numpy(x).long().cuda()
    segment_sizes = torch.tensor(segment_sizes).long().cuda()
    _ = embedding_model.embed(x.unsqueeze(0), segment_sizes.unsqueeze(0))


def _get_logps_tiered_fast(
    memory: Optional[list[PackedTensorSequences]],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    max_variant_length = max(len(v) for v in variants)
    memory = model.logits_allocate_memory(
        memory=memory,
        batch_size=batch_size,
        length=max_variant_length - 1,  # discount stop token
    )
    criteria = nn.CrossEntropyLoss(ignore_index=alphabet.mask_token, reduction="none")
    logps = []
    if pbar_position is not None:
        pbar = trange(
            0,
            len(variants),
            batch_size,
            desc=f"[{pbar_position}] decoding",
            leave=False,
            position=pbar_position,
        )
    else:
        pbar = range(0, len(variants), batch_size)
    for start_idx in pbar:
        this_variants = variants[start_idx : start_idx + batch_size]
        this_variants = pad_sequence(
            [torch.from_numpy(v).long() for v in this_variants],
            batch_first=True,
            padding_value=alphabet.mask_token,
        )
        if this_variants.size(1) < max_variant_length:
            this_variants = F.pad(
                this_variants,
                (0, max_variant_length - this_variants.size(1)),
                value=alphabet.mask_token,
            )
        assert (this_variants == alphabet.gap_token).sum() == 0
        this_variants = this_variants.cuda()
        logits = model.logits(this_variants[:, :-1], memory, preallocated_memory=True)
        targets = this_variants[:, 1:]
        score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
        logps.append(score.cpu().numpy())
    return np.hstack(logps)


def get_logps_tiered_fast(
    msa_sequences: Sequence[np.ndarray],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    if len(msa_sequences) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_sequences]).cuda()
        msa_sequences: torch.Tensor = torch.cat(
            [torch.from_numpy(s).long() for s in msa_sequences]
        ).cuda()
        memory = model.embed(
            msa_sequences.unsqueeze(0),
            segment_sizes.unsqueeze(0),
            pbar_position=pbar_position,
        )
    else:
        memory = None

    return _get_logps_tiered_fast(
        memory=memory,
        variants=variants,
        model=model,
        batch_size=batch_size,
        alphabet=alphabet,
        pbar_position=pbar_position,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="data/poet.ckpt")
    parser.add_argument(
        "--msa_a3m_path", type=str, default="data/BLAT_ECOLX_ColabFold_2202.a3m"
    )
    parser.add_argument(
        "--variants_fasta_path",
        type=str,
        default="data/BLAT_ECOLX_Jacquier_2013_variants.fasta",
    )
    parser.add_argument(
        "--output_npy_path",
        type=str,
        default="data/BLAT_ECOLX_Jacquier_2013_variants.npy",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=188257)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run only 1/15 params from the msa sampling and filtering ensemble",
    )

    args = parser.parse_args()
    args.msa_a3m_path = Path(args.msa_a3m_path)
    args.variants_fasta_path = Path(args.variants_fasta_path)
    args.output_npy_path = Path(args.output_npy_path)
    return args


@torch.inference_mode()
def main():
    args = parse_args()

    # load model
    ckpt = torch.load(args.ckpt_path)
    model = PoET(**ckpt["hyper_parameters"]["model_spec"]["init_args"])
    model.load_state_dict(
        {k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()}
    )
    del ckpt
    model = model.cuda().half().eval()
    alphabet = Uniprot21(
        include_gap=True, include_startstop=True, distinct_startstop=True
    )
    jit_warmup(model, alphabet)

    # get variants to score
    variants = [
        append_startstop(alphabet.encode(v), alphabet=alphabet)
        for v in get_seqs_from_fastalike(args.variants_fasta_path)
    ]

    # process msa
    msa_sequences = get_seqs_from_fastalike(args.msa_a3m_path)
    msa = get_encoded_msa_from_a3m_seqs(msa_sequences=msa_sequences, alphabet=alphabet)

    # score the variants
    logps = []
    if not args.debug:
        params = list(
            itertools.product(
                [6144, 12288, 24576],
                [1.0, 0.95, 0.90, 0.70, 0.50],
            )
        )
    else:
        params = [(12288, 0.95)]
    for max_tokens, max_similarity in tqdm(params, desc="ensemble"):
        sampler = MSASampler(
            method=NeighborsSampler(
                can_use_torch=False,
            ),
            max_similarity=max_similarity,
        )
        sample_idxs = sampler.get_sample_idxs(
            msa=msa,
            gap_token=alphabet.gap_token,
            seed=args.seed,
        )
        # create the sequence-of-sequences
        this_msa_sequences = sample_msa_sequences(
            get_sequence_fn=lambda ii: msa_sequences[ii]
            .upper()
            .translate(None, delete=b"-"),
            sample_idxs=sample_idxs,
            max_tokens=max_tokens,
            alphabet=alphabet,
            shuffle_seed=args.seed,
            truncate=False,
        )
        forward_logps = get_logps_tiered_fast(
            msa_sequences=this_msa_sequences,
            variants=variants,
            model=model,
            batch_size=args.batch_size,
            alphabet=alphabet,
            pbar_position=PBAR_POSITION,
        )
        backward_logps = get_logps_tiered_fast(
            msa_sequences=[np.ascontiguousarray(s[::-1]) for s in this_msa_sequences],
            variants=[np.ascontiguousarray(s[::-1]) for s in variants],
            model=model,
            batch_size=args.batch_size,
            alphabet=alphabet,
            pbar_position=PBAR_POSITION,
        )
        this_logps = (forward_logps + backward_logps) / 2
        logps.append(this_logps)
    logps = np.vstack(logps).mean(axis=0)
    np.save(args.output_npy_path, logps)


if __name__ == "__main__":
    main()
