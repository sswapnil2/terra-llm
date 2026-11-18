"""Train/validation document split and resume offsets for `stream_annotated_batches`."""

import sys
import types

from train import distill_colab as dc


class _Tok:
    eos_token_id = 0
    is_fast = False

    def encode(self, text, **_):
        # Each doc becomes 40 copies of its own id, so batches reveal which docs they came from.
        return [int(text.split("#")[1])] * 40


def _fake_datasets(monkeypatch, n_docs=5000):
    def load_dataset(path, *_, **__):
        base = 1_000_000 if "stack" in path else 1  # code ids never collide with prose ids
        return [{"text": "x" * 64 + f"#{base + i}"} for i in range(n_docs)]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))


def _doc_ids(batches):
    return {int(t) for ids, _, _ in batches for t in ids.flatten().tolist()} - {0}


def test_train_never_reads_validation_docs_and_resume_skips_consumed(monkeypatch):
    _fake_datasets(monkeypatch)
    kw = dict(seq_len=64, batch_size=2, use_edge_hints=False)

    val = dc.collect_packed_eval_batches(_Tok(), num_batches=4, **kw)
    reserve = dc.EVAL_RESERVED_DOCS

    progress: dict[str, int] = {}
    stream = dc.stream_annotated_batches(
        _Tok(), skip_prose_docs=reserve, skip_code_docs=reserve, progress=progress, **kw
    )
    first = [next(stream) for _ in range(6)]
    assert not (_doc_ids(val) & _doc_ids(first))
    assert max(_doc_ids(val) - {i for i in _doc_ids(val) if i >= 1_000_000}) <= reserve

    # Resume from the saved offsets: nothing fully consumed before is read again.
    resumed = dc.stream_annotated_batches(
        _Tok(), skip_prose_docs=progress["prose_docs"], skip_code_docs=progress["code_docs"], **kw
    )
    second = [next(resumed) for _ in range(6)]
    assert min(i for i in _doc_ids(second) if i < 1_000_000) > max(
        i for i in _doc_ids(first) if i < 1_000_000
    )
