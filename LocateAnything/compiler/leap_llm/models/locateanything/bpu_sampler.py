"""BPU-side PBD sampling primitives.

The sampler keeps the complete vocabulary.  Host code supplies a compact
history mask and one uniform value per row; the graph returns token candidates
and the probability summaries required by LocateAnything's box/ref grammar.
"""

from __future__ import annotations

import numpy as np
from hbdk4.compiler import leap


VOCAB_SIZE = 152681
PBD_ROWS = 6
TOP_CANDIDATES = 5
COORDINATE_START = 151677
COORDINATE_END = 152677
SPECIAL_TOKEN_IDS = (151668, 151672, 151669, 152678, 151645, 4064)


def _constant_ids(values: tuple[int, ...], rows: int) -> np.ndarray:
    return np.tile(np.asarray(values, dtype=np.int32), (1, rows, 1))


def build_pbd_sampler(
    logits,
    history_mask,
    random_values,
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    """Return compact BPU sampling outputs for six PBD rows.

    Output order is sampled ids, global top candidates/probabilities, the
    special-token probabilities, coordinate candidates/probabilities, and
    nucleus mass.  All sorting and cumulative probability work stays in the
    graph; no Top-K vocabulary truncation is used for sampling.
    """
    work = leap.cast_type(logits, output_type=leap.float32)
    history = leap.greater(history_mask, 0)
    adjusted = leap.where(
        leap.less(work, 0.0),
        leap.mul(work, float(repetition_penalty)),
        leap.div(work, float(repetition_penalty)),
    )
    work = leap.where(history, adjusted, work)
    probabilities = leap.softmax(leap.div(work, float(temperature)), -1)
    sorted_probabilities, sorted_ids = leap.sort(
        probabilities, dim=-1, descending=True, indices_type=leap.int32
    )
    cumulative = leap.cumsum(sorted_probabilities, -1)
    previous = leap.sub(cumulative, sorted_probabilities)
    retained = leap.less(previous, float(top_p))
    nucleus = leap.where(retained, sorted_probabilities, 0.0)
    nucleus_mass = leap.reduce_sum(nucleus, [-1], True)
    normalized_sorted = leap.div(nucleus, nucleus_mass)

    target = leap.mul(leap.reshape(random_values, [1, PBD_ROWS, 1]), 1.0)
    cumulative_nucleus = leap.cumsum(normalized_sorted, -1)
    hit = leap.cast_type(
        leap.greater_equal(cumulative_nucleus, target), output_type=leap.int32
    )
    rank = leap.reduce_argmax(hit, [-1], True, output_type=leap.int32)
    sampled_ids = leap.gather_elements(sorted_ids, rank, -1)

    zero = leap.mul(probabilities, 0.0)
    normalized = leap.scatter_elements(
        zero, sorted_ids, normalized_sorted, axis=-1, output_type=leap.float32
    )
    top_probabilities, top_ids = leap.topk(
        normalized, TOP_CANDIDATES, dim=-1, largest=True, sorted=True,
        indices_type=leap.int32,
    )
    special_ids = _constant_ids(SPECIAL_TOKEN_IDS, PBD_ROWS)
    special_probabilities = leap.gather_elements(normalized, special_ids, -1)
    coordinate_probabilities = leap.slice(
        normalized, [0, 0, COORDINATE_START],
        [1, PBD_ROWS, COORDINATE_END + 1], [1, 1, 1]
    )
    coordinate_probabilities, coordinate_offsets = leap.topk(
        coordinate_probabilities, 4, dim=-1, largest=True, sorted=True,
        indices_type=leap.int32,
    )
    coordinate_ids = leap.add(coordinate_offsets, COORDINATE_START)
    return (
        sampled_ids,
        top_ids,
        leap.cast_type(top_probabilities, output_type=leap.float16),
        leap.cast_type(special_probabilities, output_type=leap.float16),
        coordinate_ids,
        leap.cast_type(coordinate_probabilities, output_type=leap.float16),
        leap.cast_type(nucleus_mass, output_type=leap.float16),
    )
