"""Milestone 2 Verification Gates (`edge_codec/`):
1. 100% exact local variable-definition linking across a 25-file Python test suite (`tree-sitter`).
2. ONNX coreference linking + high-entropy flag (`confidence < 0.65` -> `entropy_bucket == 3`).
3. CPU latency < 2.0 ms per 512-token mixed Python + English prose snippet.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from edge_codec import (
    HIGH_ENTROPY_BUCKET,
    MODALITY_CODE,
    MODALITY_PROSE,
    OnnxProseCorefParser,
    UnifiedEdgeCodec,
)


def _edge_cli_bin() -> Path:
    edge_dir = Path(__file__).resolve().parents[1]
    cli = edge_dir / "target" / "release" / "edge-codec-cli"
    if not cli.is_file():
        subprocess.run(["cargo", "build", "--release"], cwd=str(edge_dir), check=True)
    return cli


# 25 distinct Python files covering functions, default args, closures, comprehensions, loops,
# tuple unpacking, with-as, except-as, imports, class methods, walrus operator, and shadowing.
PYTHON_25_SUITE: list[tuple[str, str, list[tuple[str, int, int]]]] = [
    (
        "f01_basic_assign.py",
        "alpha = 10\nbeta = alpha + 5\n",
        [("alpha", 1, 0)],
    ),
    (
        "f02_func_param.py",
        "def scale(val, factor):\n    res = val * factor\n    return res\n",
        [("val", 1, 0), ("factor", 1, 0), ("res", 2, 1)],
    ),
    (
        "f03_shadowing_outer.py",
        "x = 100\ndef inner(x):\n    return x + 1\ny = x + 2\n",
        [("x", 2, 1), ("x", 3, 0)],
    ),
    (
        "f04_closure_capture.py",
        "def outer(base):\n    def add(delta):\n        return base + delta\n    return add\n",
        [("base", 2, 0), ("delta", 2, 1), ("add", 3, 1)],
    ),
    (
        "f05_list_comp.py",
        "nums = [1, 2, 3]\nsquares = [n * n for n in nums if n > 0]\n",
        [("nums", 1, 0), ("n", 1, 1)],
    ),
    (
        "f06_dict_comp.py",
        "keys = ['a', 'b']\nlookup = {k: len(k) for k in keys}\n",
        [("keys", 1, 0), ("k", 1, 1)],
    ),
    (
        "f07_for_loop.py",
        "total = 0\nfor item in [1, 2, 3]:\n    total = total + item\n",
        [("total", 2, 0), ("item", 2, 1)],
    ),
    (
        "f08_tuple_unpack.py",
        "pair = (4, 9)\nleft, right = pair\ndiff = right - left\n",
        [("pair", 1, 0), ("right", 2, 1), ("left", 2, 1)],
    ),
    (
        "f09_with_as.py",
        "def read_buf(opener):\n    with opener() as handle:\n        data = handle\n    return data\n",
        [("opener", 1, 0), ("handle", 2, 1), ("data", 3, 2)],
    ),
    (
        "f10_except_as.py",
        "def safe(fn):\n    try:\n        return fn()\n    except ValueError as err:\n        return err\n",
        [("fn", 2, 0), ("err", 4, 3)],
    ),
    (
        "f11_class_method.py",
        "offset = 7\nclass Adder:\n    def apply(self, val):\n        return self.calc(val + offset)\n",
        [("self", 3, 2), ("val", 3, 2), ("offset", 3, 0)],
    ),
    (
        "f12_default_arg_outer.py",
        "seed = 42\ndef make(seed=seed):\n    return seed + 1\n",
        [("seed", 2, 1)],
    ),
    (
        "f13_import_alias.py",
        "import math as m\nfrom os import path as osp\nv = m.sqrt(4)\np = osp\n",
        [("m", 2, 0), ("osp", 3, 1)],
    ),
    (
        "f14_walrus.py",
        "items = [10, 20]\nif (count := len(items)) > 0:\n    out = count\n",
        [("items", 1, 0), ("count", 2, 1)],
    ),
    (
        "f15_lambda_scope.py",
        "x = 9\nfn = lambda x: x * 3\nz = x + fn(2)\n",
        [("x", 1, 1), ("x", 2, 0), ("fn", 2, 1)],
    ),
    (
        "f16_kwargs_call.py",
        "def helper(a, b):\n    return a - b\nc = 5\nd = 2\nout = helper(a=c, b=d)\n",
        [("helper", 4, 0), ("c", 4, 2), ("d", 4, 3)],
    ),
    (
        "f17_nested_comp.py",
        "matrix = [[1, 2], [3, 4]]\nflat = [cell for row in matrix for cell in row]\n",
        [("matrix", 1, 0), ("row", 1, 1), ("cell", 1, 1)],
    ),
    (
        "f18_recursion.py",
        "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1)\n",
        [("n", 1, 0), ("n", 2, 0), ("fib", 3, 0), ("n", 3, 0)],
    ),
    (
        "f19_nonlocal_counter.py",
        "def counter(start):\n    cur = start\n    def inc(step):\n        nonlocal cur\n        cur = cur + step\n        return cur\n    return inc\n",
        [("start", 1, 0), ("cur", 4, 1), ("step", 4, 2), ("cur", 5, 1), ("inc", 6, 2)],
    ),
    (
        "f20_typed_params.py",
        "def norm(vec: list[float], eps: float = 1e-5) -> float:\n    s = vec\n    return eps\n",
        [("vec", 1, 0), ("eps", 2, 0)],
    ),
    (
        "f21_aug_assign.py",
        "acc = 1\nstep = 3\nacc += step\nout = acc\n",
        [("acc", 2, 0), ("step", 2, 1), ("acc", 3, 0)],
    ),
    (
        "f22_attr_rhs_vs_lhs.py",
        "def update(obj, value):\n    obj.field = value\n    return obj\n",
        [("value", 1, 0), ("obj", 1, 0), ("obj", 2, 0)],
    ),
    (
        "f23_subscript_lhs.py",
        "def put(buf, idx, val):\n    buf[idx] = val\n    return buf\n",
        [("val", 1, 0), ("buf", 1, 0), ("idx", 1, 0), ("buf", 2, 0)],
    ),
    (
        "f24_reassigned_same_scope.py",
        "weight = 10\nhalf = weight / 2\nweight = 20\nfinal_w = weight + half\n",
        [("weight", 1, 0), ("weight", 3, 0), ("half", 3, 1)],
    ),
    (
        "f25_multi_func_module.py",
        "bias = 1.0\ndef f_one(u):\n    return u + bias\ndef f_two(u):\n    return f_one(u) + bias\n",
        [("u", 2, 1), ("bias", 2, 0), ("f_one", 4, 1), ("u", 4, 3), ("bias", 4, 0)],
    ),
]


def test_25_python_files_100_percent_exact_variable_binding(tmp_path: Path) -> None:
    assert len(PYTHON_25_SUITE) == 25
    cli = _edge_cli_bin()

    for fname, code, checks in PYTHON_25_SUITE:
        fpath = tmp_path / fname
        fpath.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [str(cli), "--lang", "python", "--file", str(fpath)],
            capture_output=True,
            text=True,
            check=True,
        )
        tokens = json.loads(proc.stdout)

        for var_name, use_line, expected_def_line in checks:
            matching = [
                t
                for t in tokens
                if t["text"] == var_name and t["start_line"] == use_line and not t["is_definition"]
            ]
            assert matching, f"{fname}: expected reference `{var_name}` on line {use_line}"
            for tok in matching:
                assert tok["def_line"] == expected_def_line, (
                    f"{fname}: `{var_name}` at line {use_line} bound to def_line={tok['def_line']} "
                    f"(expected {expected_def_line})"
                )
                assert tok["hint"][0] == MODALITY_CODE
                assert tok["hint"][2] == tok["def_token_index"] - tok["index"]
                assert tok["hint"][2] != 0
                if expected_def_line < use_line:
                    assert tok["hint"][2] < 0


def test_onnx_prose_coref_linking_and_low_confidence_entropy_flag() -> None:
    parser = OnnxProseCorefParser()

    # 1. Clear non-ambiguous coreference: "Sarah" <- "She", "telescope" <- "it"
    clear_text = "Sarah bought a telescope yesterday. She tested it at night."
    _, clear_hints, clear_links = parser.parse_with_details(clear_text)
    link_by_mention = {lnk["mention"].lower(): lnk for lnk in clear_links}
    assert link_by_mention["she"]["antecedent"].lower() == "sarah"
    assert link_by_mention["she"]["confidence"] >= 0.65
    assert not link_by_mention["she"]["high_entropy"]
    assert link_by_mention["it"]["antecedent"].lower() == "telescope"
    assert link_by_mention["it"]["confidence"] >= 0.65
    assert clear_hints[link_by_mention["she"]["mention_tok"], 0] == MODALITY_PROSE
    assert clear_hints[link_by_mention["she"]["mention_tok"], 2] < 0

    # 2. Ambiguous competing antecedents ("Alice" and "Brenda" both female singular):
    #    confidence drops below 0.65 and triggers HIGH_ENTROPY_BUCKET (3)!
    ambig_text = "Alice thanked Brenda after she solved the puzzle."
    _, ambig_hints, ambig_links = parser.parse_with_details(ambig_text)
    ambig_she = next(lnk for lnk in ambig_links if lnk["mention"].lower() == "she")
    assert ambig_she["confidence"] < 0.65
    assert ambig_she["high_entropy"] is True
    assert ambig_hints[ambig_she["mention_tok"], 3] == HIGH_ENTROPY_BUCKET


def test_mixed_512_token_cpu_latency_under_2ms() -> None:
    codec = UnifiedEdgeCodec()

    prose_block = (
        "Sarah built a compiler yesterday. She tested it on mixed Python modules, "
        "and Alice thanked Brenda after she reviewed the benchmark logs. "
    ) * 4
    code_block = "\n".join(
        f"def layer_step_{i}(hidden_{i}, scale_{i}):\n    out_{i} = hidden_{i} * scale_{i}\n    return out_{i}\n"
        for i in range(25)
    )
    mixed_doc = f"{prose_block}\n```python\n{code_block}\n```\n{prose_block}"

    warm = codec.encode_mixed(mixed_doc, max_tokens=512)
    assert warm.shape == (512, 4)
    assert warm.dtype == np.int32
    assert np.any(warm[:, 0] == MODALITY_PROSE)
    assert np.any(warm[:, 0] == MODALITY_CODE)

    latencies_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        res = codec.encode_mixed(mixed_doc, max_tokens=512)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert res.shape[0] == 512

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    print(f"[UnifiedEdgeCodec 512-tok mixed CPU] mean={mean_ms:.4f} ms, median={median_ms:.4f} ms")
    assert mean_ms < 2.0, f"Expected mean latency < 2.0 ms, got {mean_ms:.4f} ms"
    assert median_ms < 2.0, f"Expected median latency < 2.0 ms, got {median_ms:.4f} ms"


def test_bpe_span_alignment_preserves_bpe_relative_def_offsets() -> None:
    codec = UnifiedEdgeCodec()

    # Case 1: Mixed prose + fenced Python where multi-char identifiers are BPE-split into subwords.
    sample = "Sarah checked the log and she ran:\n```python\ndef compute_norm(accumulator_value):\n    return accumulator_value + 1\n```"
    # Simulated BPE pieces with exact character offsets into `sample`, including special (0,0) BOS/EOS
    # and split subwords (`"accumulator"` + `"_value"`).
    bpe_pieces = [
        "<bos>",
        "Sarah", " checked", " the", " log", " and", " she", " ran", ":",
        "\n```python\n",
        "def", " compute", "_norm", "(", "accumulator", "_value", "):\n    ",
        "return", " accumulator", "_value", " +", " 1", "\n```",
        "<eos>",
    ]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for p in bpe_pieces:
        if p in ("<bos>", "<eos>"):
            offsets.append((0, 0))
        else:
            idx = sample.index(p, cursor)
            # Strip leading whitespace on `" checked"` etc., matching HuggingFace byte-BPE offsets
            stripped_lead = len(p) - len(p.lstrip())
            offsets.append((idx + stripped_lead, idx + len(p)))
            cursor = idx + len(p)

    aligned = codec.encode_aligned_to_bpe(sample, offsets)
    assert aligned.shape == (len(bpe_pieces), 4)
    assert aligned.dtype == np.int32

    # Special (0,0) BOS/EOS tokens stay zeroed
    assert np.array_equal(aligned[0], [0, 0, 0, 0])
    assert np.array_equal(aligned[-1], [0, 0, 0, 0])

    # Code use of split identifier `accumulator` + `_value` points back to def start BPE index!
    def_bpe_idx = bpe_pieces.index("accumulator")
    use_bpe_idx_0 = bpe_pieces.index(" accumulator")
    use_bpe_idx_1 = len(bpe_pieces) - 1 - bpe_pieces[::-1].index("_value")
    assert aligned[use_bpe_idx_0, 0] == MODALITY_CODE
    assert aligned[use_bpe_idx_0, 2] == def_bpe_idx - use_bpe_idx_0
    assert aligned[use_bpe_idx_1, 0] == MODALITY_CODE
    assert aligned[use_bpe_idx_1, 2] == def_bpe_idx - use_bpe_idx_1

    # Prose coreference `she` -> `Sarah` points to exact BPE index of `Sarah`
    sarah_idx = bpe_pieces.index("Sarah")
    she_idx = bpe_pieces.index(" she")
    assert aligned[she_idx, 0] == MODALITY_PROSE
    assert aligned[she_idx, 2] == sarah_idx - she_idx

    # Non-ASCII UTF-8 check: multi-byte dash before code doesn't shift char spans
    utf8_src = "x_val = 10  # — note\ny_val = x_val + 2\n"
    x_def = utf8_src.index("x_val")
    y_pos = utf8_src.index("y_val")
    x_use = utf8_src.index("x_val", y_pos)
    mini_offsets = [
        (x_def, x_def + 5),
        (utf8_src.index("10"), utf8_src.index("10") + 2),
        (y_pos, y_pos + 5),
        (x_use, x_use + 5),
    ]
    utf8_aligned = codec.encode_aligned_to_bpe(utf8_src, mini_offsets, is_code=True)
    assert utf8_aligned[3, 0] == MODALITY_CODE
    assert utf8_aligned[3, 2] == -3

    # Truncated forward-referenced list comprehension (`max_code_tokens=3` drops `for item_var`)
    # must not raise IndexError on `def_token_index >= count`.
    comp_src = "[item_var + 1 for item_var in range(5)]"
    tiny_codec = UnifiedEdgeCodec(max_code_tokens=3)
    comp_offsets = [(comp_src.index("item_var"), comp_src.index("item_var") + 8)]
    comp_aligned = tiny_codec.encode_aligned_to_bpe(comp_src, comp_offsets, is_code=True)
    assert comp_aligned.shape == (1, 4)
    assert comp_aligned[0, 2] == 0
