"""Recurrent neural network architectures including language models and seq2seq variants."""

from __future__ import annotations

from .attention import (
    AttentionDecoder,
    AttentionEncoder,
    AttentionSeq2seq,
)
from .attention_peeky import (
    AttentionPeekyDecoder,
    AttentionPeekySeq2seq,
)
from .lm import (
    BetterRnnlm,
    Rnnlm,
    TiedRnnlm,
    VanillaRnnlm,
)
from .peeky import (
    PeekyDecoder,
    PeekySeq2seq,
)
from .sampling import (
    _host_sampled_ids as _host_sampled_ids,
)
from .sampling import (
    _stack_sampled_ids_device as _stack_sampled_ids_device,
)
from .seq2seq import (
    Decoder,
    Encoder,
    Seq2seq,
)

__all__ = [
    "AttentionDecoder",
    "AttentionEncoder",
    "AttentionPeekyDecoder",
    "AttentionPeekySeq2seq",
    "AttentionSeq2seq",
    "BetterRnnlm",
    "Decoder",
    "Encoder",
    "PeekyDecoder",
    "PeekySeq2seq",
    "Rnnlm",
    "Seq2seq",
    "TiedRnnlm",
    "VanillaRnnlm",
]
