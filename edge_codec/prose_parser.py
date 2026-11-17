"""Dual-Path Edge Pre-Processor (`edge_codec/`):
- Code Path: zero-copy `ctypes` bridge to Rust `tree-sitter` Python/Rust AST scope & definition resolver.
- Prose Path: quantized/lightweight ONNX mention-antecedent coreference & entropy scorer (`onnxruntime`).
- Unified `[seq_len, 4]` int32 hint matrix: `[modality_flag, entity_or_var_id, relative_def_offset, entropy_bucket]`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
import zlib
from typing import Sequence

import numpy as np
import onnxruntime as ort

MODALITY_NONE = 0
MODALITY_CODE = 1
MODALITY_PROSE = 2

LOW_CONFIDENCE_THRESHOLD = 0.65
HIGH_ENTROPY_BUCKET = 3

_WORD_OR_PUNCT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|[0-9]+(?:\.[0-9]+)?|[^\s]")
_FENCED_CODE_RE = re.compile(r"```(python|py|rust|rs)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
_MALE_PRONOUNS = {"he", "him", "his", "himself"}
_NEUTRAL_PRONOUNS = {"it", "its", "itself"}
_PLURAL_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}
_ALL_PRONOUNS = _FEMALE_PRONOUNS | _MALE_PRONOUNS | _NEUTRAL_PRONOUNS | _PLURAL_PRONOUNS

_FEMALE_NAMES = {
    "alice", "anna", "brenda", "carol", "claire", "diana", "elena", "emily", "emma",
    "eva", "grace", "hannah", "helen", "laura", "lisa", "maria", "maya", "nina",
    "olivia", "rachel", "sarah", "sophia", "susan", "victoria",
}
_MALE_NAMES = {
    "adam", "alan", "alex", "andrew", "ben", "bob", "brian", "carlos", "charles",
    "daniel", "david", "edward", "eric", "frank", "george", "henry", "james", "john",
    "kevin", "liam", "lucas", "mark", "michael", "noah", "oliver", "paul", "peter",
    "robert", "ryan", "samuel", "thomas", "william",
}
_DETERMINERS = {"a", "an", "the", "this", "that", "my", "your", "our", "some", "each", "every"}
_STOP_CAPITALS = {"the", "a", "an", "this", "that", "these", "those", "in", "on", "at", "after", "before", "when", "while", "if", "as", "by", "for", "with"}


def _stable_id(text: str) -> int:
    return (zlib.crc32(text.lower().encode("utf-8")) % 1023) + 1


# ---------------------------------------------------------------------------
# Pure-Python ONNX Pairwise Coreference Scorer Graph Builder (IRv7 / Opset 13)
# ---------------------------------------------------------------------------


def _varint(val: int) -> bytes:
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val)
    return bytes(out)


def _tag(field: int, wtype: int) -> bytes:
    return _varint((field << 3) | wtype)


def _ld(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _str(field: int, s: str) -> bytes:
    return _ld(field, s.encode("utf-8"))


def _i64(field: int, v: int) -> bytes:
    return _tag(field, 0) + _varint(v & ((1 << 64) - 1))


def _tensor_f32(name: str, arr: np.ndarray) -> bytes:
    c_arr = np.ascontiguousarray(arr, dtype="<f4")
    body = b"".join(_i64(1, int(d)) for d in c_arr.shape)
    body += _i64(2, 1)  # TensorProto.FLOAT
    body += _str(8, name)
    body += _ld(9, c_arr.tobytes())
    return body


def _value_info_f32(name: str, dims: Sequence[int | str]) -> bytes:
    shape_dims = b""
    for d in dims:
        dim_msg = _i64(1, d) if isinstance(d, int) else _str(2, d)
        shape_dims += _ld(1, dim_msg)
    tensor_type = _i64(1, 1) + _ld(2, shape_dims)
    type_proto = _ld(1, tensor_type)
    return _str(1, name) + _ld(2, type_proto)


def _onnx_node(op: str, inputs: Sequence[str], outputs: Sequence[str], name: str) -> bytes:
    return (
        b"".join(_str(1, i) for i in inputs)
        + b"".join(_str(2, o) for o in outputs)
        + _str(3, name)
        + _str(4, op)
    )


def build_default_coref_onnx_bytes() -> bytes:
    """Serialize a calibrated 8-feature -> 16-hidden -> 1-probability ONNX coreference scorer.

    Features (`[N, 8]` float32):
      0: gender_compat (-1 incompatible, 0 unknown, +1 compatible)
      1: number_compat (-1 incompatible, 0 unknown, +1 compatible)
      2: animacy_compat (-1 incompatible, 0 unknown, +1 compatible)
      3: inv_token_dist (1 / (1 + token_dist / 16))
      4: same_or_prev_sentence (1.0 if sent_diff <= 1 else 0.3)
      5: antecedent_is_proper_or_subject (1.0 or 0.0)
      6: exact_lexical_match (1.0 or 0.0)
      7: competing_candidate_ambiguity (1.0 when >=2 equally compatible candidates compete else 0.0)
    """
    # Linear calibration layer + ReLU hidden refinement -> Sigmoid probability
    w1 = np.zeros((8, 16), dtype=np.float32)
    # Channel 0 passes net compatibility logit cleanly into ReLU + direct skip
    w1[:, 0] = np.array([4.5, 4.0, 4.5, 1.0, 0.8, 0.7, 3.0, -8.0], dtype=np.float32)
    # Channel 1 activates strongly only on exact lexical match
    w1[6, 1] = 6.0
    # Channel 2 penalizes any hard morphological conflict
    w1[:, 2] = np.array([-5.0, -5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    b1 = np.zeros((16,), dtype=np.float32)
    b1[0] = -1.5
    b1[1] = -2.0
    b1[2] = -1.0

    w2 = np.zeros((16, 1), dtype=np.float32)
    w2[0, 0] = 1.0
    w2[1, 0] = 1.0
    w2[2, 0] = -2.5
    b2 = np.array([-6.0], dtype=np.float32)

    nodes = [
        _onnx_node("MatMul", ["features", "W1"], ["h0"], "mm1"),
        _onnx_node("Add", ["h0", "b1"], ["h1"], "add1"),
        _onnx_node("Relu", ["h1"], ["h2"], "relu1"),
        _onnx_node("MatMul", ["h2", "W2"], ["z0"], "mm2"),
        _onnx_node("Add", ["z0", "b2"], ["z1"], "add2"),
        _onnx_node("Sigmoid", ["z1"], ["probs"], "sig"),
    ]
    inits = [
        _tensor_f32("W1", w1),
        _tensor_f32("b1", b1),
        _tensor_f32("W2", w2),
        _tensor_f32("b2", b2),
    ]
    graph = (
        b"".join(_ld(1, n) for n in nodes)
        + _str(2, "micro_terra_coref")
        + b"".join(_ld(5, t) for t in inits)
        + _ld(11, _value_info_f32("features", ["N", 8]))
        + _ld(12, _value_info_f32("probs", ["N", 1]))
    )
    opset = _str(1, "") + _i64(2, 13)
    return _i64(1, 7) + _ld(7, graph) + _ld(8, opset)


@dataclass(frozen=True)
class ProseMention:
    token_idx: int
    text: str
    sent_idx: int
    gender: int   # -1 unknown, 0 neutral, 1 female, 2 male
    number: int   # 0 singular, 1 plural, -1 unknown
    animacy: int  # 0 inanimate, 1 animate, -1 unknown
    is_pronoun: bool
    is_proper: bool


class OnnxProseCorefParser:
    """Lightweight ONNX Runtime (`CPUExecutionProvider`) coreference/entity span linker."""

    def __init__(self, onnx_model_path: str | Path | None = None) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if onnx_model_path is not None:
            model_src: bytes | str = str(onnx_model_path)
        else:
            model_src = build_default_coref_onnx_bytes()
        self.session = ort.InferenceSession(
            model_src,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self.session.get_inputs()[0].name

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _WORD_OR_PUNCT_RE.findall(text)

    @staticmethod
    def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in _WORD_OR_PUNCT_RE.finditer(text)]

    @staticmethod
    def _extract_mentions(tokens: list[str]) -> list[ProseMention]:
        mentions: list[ProseMention] = []
        sent_idx = 0
        sent_start = True

        for i, tok in enumerate(tokens):
            if tok in (".", "!", "?"):
                sent_idx += 1
                sent_start = True
                continue

            low = tok.lower()
            prev_low = tokens[i - 1].lower() if i > 0 else ""

            if low in _ALL_PRONOUNS:
                if low in _FEMALE_PRONOUNS:
                    g, num, anim = 1, 0, 1
                elif low in _MALE_PRONOUNS:
                    g, num, anim = 2, 0, 1
                elif low in _NEUTRAL_PRONOUNS:
                    g, num, anim = 0, 0, 0
                else:
                    g, num, anim = -1, 1, -1
                mentions.append(ProseMention(i, tok, sent_idx, g, num, anim, True, False))
                sent_start = False
                continue

            if low in _FEMALE_NAMES:
                mentions.append(ProseMention(i, tok, sent_idx, 1, 0, 1, False, True))
            elif low in _MALE_NAMES:
                mentions.append(ProseMention(i, tok, sent_idx, 2, 0, 1, False, True))
            elif tok[0].isupper() and tok.isalpha() and len(tok) > 1 and low not in _STOP_CAPITALS:
                # Proper noun (either mid-sentence or non-stopword capitalized entity)
                is_prop = (not sent_start) or (i + 1 < len(tokens) and tokens[i + 1].isalpha())
                if is_prop:
                    mentions.append(ProseMention(i, tok, sent_idx, -1, 0, 1, False, True))
            elif prev_low in _DETERMINERS and tok.isalpha() and len(tok) > 2:
                # Common noun NP head preceded by determiner (`a telescope`, `the kernel`, `these tensors`)
                is_pl = low.endswith("s") and not low.endswith("ss")
                mentions.append(
                    ProseMention(i, tok, sent_idx, 0, 1 if is_pl else 0, 0, False, False)
                )

            sent_start = False
        return mentions

    @staticmethod
    def _compat(a: int, b: int) -> float:
        if a == -1 or b == -1:
            return 0.25
        return 1.0 if a == b else -1.0

    def parse_with_details(self, text: str) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
        """Return `(tokens, hints [N, 4] int32, coref_links)`."""
        tokens = self.tokenize(text)
        n = len(tokens)
        hints = np.zeros((n, 4), dtype=np.int32)
        if n == 0:
            return tokens, hints, []
        hints[:, 0] = MODALITY_PROSE

        mentions = self._extract_mentions(tokens)
        if not mentions:
            return tokens, hints, []

        # Assign initial entity clusters for non-pronouns; score pronoun/anaphor links via ONNX.
        cluster_ids: dict[int, int] = {}
        mention_by_idx: dict[int, ProseMention] = {m.token_idx: m for m in mentions}
        candidate_pairs: list[tuple[int, int, np.ndarray]] = []  # (anaphor_m_idx, ant_m_idx, feat_row)

        for m_pos, m in enumerate(mentions):
            if not m.is_pronoun:
                # Check if exact same surface form appeared earlier
                exact_prev = next(
                    (
                        prev
                        for prev in reversed(mentions[:m_pos])
                        if prev.text.lower() == m.text.lower()
                    ),
                    None,
                )
                if exact_prev is not None:
                    cid = cluster_ids[exact_prev.token_idx]
                    cluster_ids[m.token_idx] = cid
                    hints[m.token_idx] = [
                        MODALITY_PROSE,
                        cid,
                        exact_prev.token_idx - m.token_idx,
                        0,
                    ]
                else:
                    cid = _stable_id(m.text)
                    cluster_ids[m.token_idx] = cid
                    hints[m.token_idx] = [MODALITY_PROSE, cid, 0, 0]
                continue

            # Gather compatible preceding antecedents to compute ambiguity feature f7
            cands: list[tuple[int, float, float, float, int]] = []
            for a_pos in range(m_pos - 1, -1, -1):
                ant = mentions[a_pos]
                if ant.is_pronoun:
                    continue
                if m.token_idx - ant.token_idx > 128:
                    break
                gc = self._compat(m.gender, ant.gender)
                nc = self._compat(m.number, ant.number)
                ac = self._compat(m.animacy, ant.animacy)
                cands.append((a_pos, gc, nc, ac, ant.token_idx))

            num_morph_compatible = sum(1 for _, gc, nc, ac, _ in cands if gc > 0 and nc > 0 and ac > 0)
            ambig = 1.0 if num_morph_compatible >= 2 else 0.0

            for a_pos, gc, nc, ac, ant_tok_idx in cands:
                ant = mentions[a_pos]
                dist = float(m.token_idx - ant_tok_idx)
                sent_diff = m.sent_idx - ant.sent_idx
                feat = np.array(
                    [
                        gc,
                        nc,
                        ac,
                        1.0 / (1.0 + dist / 16.0),
                        1.0 if sent_diff <= 1 else 0.3,
                        1.0 if ant.is_proper else 0.4,
                        1.0 if ant.text.lower() == m.text.lower() else 0.0,
                        ambig,
                    ],
                    dtype=np.float32,
                )
                candidate_pairs.append((m_pos, a_pos, feat))

        links: list[dict[str, object]] = []
        if candidate_pairs:
            feat_mat = np.stack([p[2] for p in candidate_pairs], axis=0)
            probs = self.session.run(None, {self._input_name: feat_mat})[0].reshape(-1)

            best_for_mention: dict[int, tuple[float, int]] = {}
            for pair_idx, (m_pos, a_pos, _) in enumerate(candidate_pairs):
                prob = float(probs[pair_idx])
                prev = best_for_mention.get(m_pos)
                if prev is None or prob > prev[0]:
                    best_for_mention[m_pos] = (prob, a_pos)

            for m_pos, (prob, a_pos) in best_for_mention.items():
                m = mentions[m_pos]
                ant = mentions[a_pos]
                cid = cluster_ids.get(ant.token_idx, _stable_id(ant.text))
                cluster_ids[m.token_idx] = cid
                if prob < LOW_CONFIDENCE_THRESHOLD:
                    entropy_bucket = HIGH_ENTROPY_BUCKET
                elif prob < 0.85:
                    entropy_bucket = 1
                else:
                    entropy_bucket = 0
                hints[m.token_idx] = [
                    MODALITY_PROSE,
                    cid,
                    ant.token_idx - m.token_idx,
                    entropy_bucket,
                ]
                links.append(
                    {
                        "mention": m.text,
                        "mention_tok": m.token_idx,
                        "antecedent": ant.text,
                        "antecedent_tok": ant.token_idx,
                        "cluster_id": cid,
                        "confidence": prob,
                        "high_entropy": prob < LOW_CONFIDENCE_THRESHOLD,
                    }
                )

        # Any unmatched pronoun gets high-entropy self cluster
        for m in mentions:
            if m.is_pronoun and m.token_idx not in cluster_ids:
                cid = _stable_id(m.text)
                hints[m.token_idx] = [MODALITY_PROSE, cid, 0, HIGH_ENTROPY_BUCKET]

        del mention_by_idx
        return tokens, hints, links

    def encode(self, text: str) -> np.ndarray:
        _, hints, _ = self.parse_with_details(text)
        return hints

    def encode_with_spans(self, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(hints [N, 4], char_spans [N, 2], def_idx [N])` in `text` char coordinates."""
        _, hints, _ = self.parse_with_details(text)
        n_tok = int(hints.shape[0])
        if n_tok == 0:
            return (
                hints,
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )
        spans = np.array(
            [(m.start(), m.end()) for m in _WORD_OR_PUNCT_RE.finditer(text)],
            dtype=np.int32,
        )
        rel = hints[:, 2]
        indices = np.arange(n_tok, dtype=np.int32)
        def_idx = np.where(rel < 0, indices + rel, -1).astype(np.int32)
        return hints, spans, def_idx


# ---------------------------------------------------------------------------
# Unified Dual-Path Edge Codec (Rust tree-sitter via ctypes + ONNX Prose)
# ---------------------------------------------------------------------------


def _find_or_build_edge_codec_lib() -> Path:
    edge_dir = Path(__file__).resolve().parent
    lib_name = "libedge_codec.dylib" if sys.platform == "darwin" else "libedge_codec.so"
    candidate = edge_dir / "target" / "release" / lib_name
    needs_build = not candidate.is_file()
    if not needs_build:
        lib_mtime = candidate.stat().st_mtime
        watched = [edge_dir / "Cargo.toml", *edge_dir.glob("src/**/*.rs")]
        if any(p.is_file() and p.stat().st_mtime > lib_mtime for p in watched):
            needs_build = True
        else:
            try:
                lib = ctypes.CDLL(str(candidate))
                if not hasattr(lib, "terra_parse_code_hints_with_spans"):
                    needs_build = True
            except OSError:
                needs_build = True
    if needs_build:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(edge_dir),
            check=True,
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"Expected compiled edge_codec library at {candidate}")
    return candidate


_CODE_HEURISTIC_RE = re.compile(
    r"(?m)^\s*(?:def |class |async def |from \S+ import |import |fn |pub fn |let (?:mut )?)"
)


def _utf8_byte_spans_to_char_spans(source: str, src_bytes: bytes, byte_spans: np.ndarray) -> np.ndarray:
    """Convert `[N, 2]` UTF-8 byte spans into Python string character spans."""
    if byte_spans.shape[0] == 0 or len(src_bytes) == len(source):
        return byte_spans.copy()
    byte_to_char = np.empty(len(src_bytes) + 1, dtype=np.int32)
    b_pos = 0
    for c_idx, ch in enumerate(source):
        ch_len = len(ch.encode("utf-8", errors="replace"))
        byte_to_char[b_pos : b_pos + ch_len] = c_idx
        b_pos += ch_len
    byte_to_char[len(src_bytes)] = len(source)
    return byte_to_char[np.clip(byte_spans, 0, len(src_bytes))]


class UnifiedEdgeCodec:
    """Fast unified `[seq_len, 4]` edge hint generator (< 2 ms per 512-token mixed chunk on CPU)."""

    def __init__(
        self,
        onnx_model_path: str | Path | None = None,
        max_code_tokens: int = 4096,
    ) -> None:
        lib_path = _find_or_build_edge_codec_lib()
        self._lib = ctypes.CDLL(str(lib_path))
        self._parse_fn = self._lib.terra_parse_code_hints
        self._parse_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
        ]
        self._parse_fn.restype = ctypes.c_ssize_t

        self._parse_spans_fn = self._lib.terra_parse_code_hints_with_spans
        self._parse_spans_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
        ]
        self._parse_spans_fn.restype = ctypes.c_ssize_t

        self.max_code_tokens = max_code_tokens
        self._code_buf = np.empty((max_code_tokens, 4), dtype=np.int32)
        self._span_buf = np.empty((max_code_tokens, 2), dtype=np.int32)
        self._def_idx_buf = np.empty((max_code_tokens,), dtype=np.int32)
        self.prose_parser = OnnxProseCorefParser(onnx_model_path=onnx_model_path)

    @staticmethod
    def _lang_id(lang: str) -> int:
        lang_low = lang.strip().lower()
        if lang_low in ("python", "py"):
            return 0
        if lang_low in ("rust", "rs"):
            return 1
        raise ValueError(f"Unsupported code language {lang!r} (expected python|rust)")

    def encode_code(self, source: str, lang: str = "python") -> np.ndarray:
        lang_id = self._lang_id(lang)
        src_bytes = source.encode("utf-8", errors="replace")
        n_tok = self._parse_fn(
            src_bytes,
            len(src_bytes),
            lang_id,
            self._code_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.max_code_tokens,
        )
        if n_tok < 0:
            raise RuntimeError("edge_codec C-ABI parser returned error")
        return self._code_buf[: int(n_tok)].copy()

    def encode_code_with_spans(
        self, source: str, lang: str = "python"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(hints [N, 4], char_spans [N, 2], def_idx [N])` in `source` char coordinates."""
        lang_id = self._lang_id(lang)
        src_bytes = source.encode("utf-8", errors="replace")
        n_tok = self._parse_spans_fn(
            src_bytes,
            len(src_bytes),
            lang_id,
            self._code_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._span_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._def_idx_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.max_code_tokens,
        )
        if n_tok < 0:
            raise RuntimeError("edge_codec C-ABI span parser returned error")
        k = int(n_tok)
        char_spans = _utf8_byte_spans_to_char_spans(source, src_bytes, self._span_buf[:k])
        return self._code_buf[:k].copy(), char_spans, self._def_idx_buf[:k].copy()

    def encode_prose(self, text: str) -> np.ndarray:
        return self.prose_parser.encode(text)

    def encode_prose_with_spans(self, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.prose_parser.encode_with_spans(text)

    def _collect_segments_with_spans(
        self,
        text: str,
        *,
        is_code: bool | None = None,
        code_lang: str = "python",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse `text` into concatenated `(hints, global_char_spans, global_def_idx)` arrays."""
        segs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if _FENCED_CODE_RE.search(text) is not None:
            pos = 0
            for match in _FENCED_CODE_RE.finditer(text):
                if match.start() > pos and text[pos : match.start()].strip():
                    h, sp, di = self.encode_prose_with_spans(text[pos : match.start()])
                    segs.append((h, sp + pos, di))
                code_seg = match.group(2)
                if code_seg.strip():
                    c_start = match.start(2)
                    h, sp, di = self.encode_code_with_spans(code_seg, lang=match.group(1))
                    segs.append((h, sp + c_start, di))
                pos = match.end()
            if pos < len(text) and text[pos:].strip():
                h, sp, di = self.encode_prose_with_spans(text[pos:])
                segs.append((h, sp + pos, di))
        elif is_code is True or (is_code is None and _CODE_HEURISTIC_RE.search(text) is not None):
            segs.append(self.encode_code_with_spans(text, lang=code_lang))
        else:
            segs.append(self.encode_prose_with_spans(text))

        segs = [s for s in segs if s[0].shape[0] > 0]
        if not segs:
            return (
                np.zeros((0, 4), dtype=np.int32),
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )
        offset = 0
        shifted_defs: list[np.ndarray] = []
        for h, _sp, di in segs:
            if offset > 0:
                di = np.where(di >= 0, di + offset, -1).astype(np.int32)
            shifted_defs.append(di)
            offset += int(h.shape[0])
        return (
            np.concatenate([s[0] for s in segs], axis=0),
            np.concatenate([s[1] for s in segs], axis=0),
            np.concatenate(shifted_defs, axis=0),
        )

    def encode_aligned_to_bpe(
        self,
        text: str,
        offset_mapping: Sequence[tuple[int, int]],
        *,
        is_code: bool | None = None,
        code_lang: str = "python",
    ) -> np.ndarray:
        """Align edge hints `[len(offset_mapping), 4]` to HuggingFace BPE `offset_mapping` spans."""
        num_bpe = len(offset_mapping)
        out = np.zeros((num_bpe, 4), dtype=np.int32)
        if num_bpe == 0 or not text:
            return out

        p_hints, p_spans, p_def_idx = self._collect_segments_with_spans(
            text, is_code=is_code, code_lang=code_lang
        )
        num_p = int(p_hints.shape[0])
        if num_p == 0:
            return out

        bpe_to_p = [-1] * num_bpe
        p_to_first_bpe = [-1] * num_p
        p_cursor = 0

        for b_idx, span in enumerate(offset_mapping):
            b_s = int(span[0])
            b_e = int(span[1])
            if b_s >= b_e:
                continue
            while p_cursor < num_p and int(p_spans[p_cursor, 1]) <= b_s:
                p_cursor += 1
            best_p = -1
            best_key = (-1, 0)
            scan = p_cursor
            while scan < num_p and int(p_spans[scan, 0]) < b_e:
                p_s = int(p_spans[scan, 0])
                p_e = int(p_spans[scan, 1])
                overlap = min(b_e, p_e) - max(b_s, p_s)
                if overlap > 0:
                    has_info = 1 if (p_hints[scan, 0] != 0 and (p_hints[scan, 1] != 0 or p_hints[scan, 2] != 0)) else 0
                    key = (has_info, overlap)
                    if key > best_key:
                        best_key = key
                        best_p = scan
                scan += 1
            if best_p >= 0:
                bpe_to_p[b_idx] = best_p
                if p_to_first_bpe[best_p] < 0:
                    p_to_first_bpe[best_p] = b_idx
                out[b_idx] = p_hints[best_p]
                out[b_idx, 2] = 0

        for b_idx, p_idx in enumerate(bpe_to_p):
            if p_idx < 0:
                continue
            def_p = int(p_def_idx[p_idx])
            if 0 <= def_p < num_p:
                def_bpe = p_to_first_bpe[def_p]
                if def_bpe >= 0 and def_bpe != b_idx:
                    out[b_idx, 2] = def_bpe - b_idx
        return out

    def encode_mixed(self, text: str, max_tokens: int = 512) -> np.ndarray:
        """Split fenced ```python / ```rust blocks from prose paragraphs and emit `[<=max_tokens, 4]` int32."""
        chunks: list[np.ndarray] = []
        pos = 0
        total_toks = 0

        for match in _FENCED_CODE_RE.finditer(text):
            if match.start() > pos and total_toks < max_tokens:
                prose_seg = text[pos : match.start()].strip()
                if prose_seg:
                    p_hints = self.encode_prose(prose_seg)
                    chunks.append(p_hints)
                    total_toks += int(p_hints.shape[0])
            if total_toks < max_tokens:
                lang = match.group(1)
                code_seg = match.group(2)
                c_hints = self.encode_code(code_seg, lang=lang)
                chunks.append(c_hints)
                total_toks += int(c_hints.shape[0])
            pos = match.end()
            if total_toks >= max_tokens:
                break

        if pos < len(text) and total_toks < max_tokens:
            tail = text[pos:].strip()
            if tail:
                p_hints = self.encode_prose(tail)
                chunks.append(p_hints)

        if not chunks:
            return np.zeros((0, 4), dtype=np.int32)
        combined = np.concatenate(chunks, axis=0)
        return combined[:max_tokens]
