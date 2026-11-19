"""Package the local GSM8K story views for the Hugging Face Dataset Viewer."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = Path("data/gsm8k-luna-100k-shared-views")
OUTPUT = Path("outputs/gsm8k-story-views-100k-shared-huggingface")

TEXT = pa.string()
TEXT_LIST = pa.list_(TEXT)
BINDINGS = pa.list_(pa.struct([("slot_id", TEXT), ("value", TEXT)]))
STEP_VALUES = pa.list_(
    pa.struct(
        [
            ("step_id", TEXT),
            ("operands", TEXT_LIST),
            ("result", TEXT),
            ("structural_signature", TEXT),
        ]
    )
)
NUMBERS = pa.list_(
    pa.struct([(name, TEXT) for name in ("id", "source", "min", "max", "step")])
)
CHOICES = pa.list_(pa.struct([("id", TEXT), ("values", TEXT_LIST)]))
STEPS = pa.list_(
    pa.struct([("id", TEXT), ("op", TEXT), ("args", TEXT_LIST), ("explanation", TEXT)])
)
CONSTRAINTS = pa.list_(pa.struct([("op", TEXT), ("args", TEXT_LIST)]))
GRAPH = pa.struct(
    [
        ("question", TEXT),
        ("numbers", NUMBERS),
        ("choices", CHOICES),
        ("steps", STEPS),
        ("constraints", CONSTRAINTS),
        ("answer", TEXT),
    ]
)
SCHEMAS = {
    "views": pa.schema(
        [
            ("question", TEXT),
            ("answer", TEXT),
            ("answer_value", TEXT),
            ("view_id", TEXT),
            ("base_id", TEXT),
            ("template_id", TEXT),
            ("variant_id", TEXT),
            ("text_setting", pa.int8()),
            ("numeric_group_id", TEXT),
            ("numeric_setting", pa.int8()),
            ("numeric_inputs", BINDINGS),
            ("text_inputs", BINDINGS),
            ("solution_steps", STEP_VALUES),
        ]
    ),
    "templates": pa.schema(
        [
            ("question_template", TEXT),
            ("template_id", TEXT),
            ("base_id", TEXT),
            ("variant_id", TEXT),
            ("premise", TEXT),
            ("solution_step_ids", TEXT_LIST),
            (
                "numeric_map",
                pa.list_(pa.struct([("base_slot_id", TEXT), ("story_slot_id", TEXT)])),
            ),
            ("template", GRAPH),
        ]
    ),
}


def rows(source, name):
    with (source / f"{name}.jsonl").open(encoding="utf-8") as handle:
        yield from map(json.loads, handle)


def view_record(row):
    result = {**row, "answer_value": row["answer"].rsplit("####", 1)[1].strip()}
    for column in ("numeric_inputs", "text_inputs"):
        result[column] = [
            {"slot_id": key, "value": value} for key, value in row[column].items()
        ]
    return result


def template_record(row):
    graph = row["template"]
    mapping = row.get(
        "numeric_map", {slot["id"]: slot["id"] for slot in graph["numbers"]}
    )
    return {
        **row,
        "question_template": graph["question"],
        "numeric_map": [
            {"base_slot_id": key, "story_slot_id": value}
            for key, value in mapping.items()
        ],
    }


def package(source, output, name, transform, expected):
    dest = output / "data" / name / "train-00000-of-00001.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".parquet.tmp")
    schema = SCHEMAS[name]
    records = map(transform, rows(source, name))
    with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
        while batch := list(islice(records, 5000)):
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    assert pq.ParquetFile(temp).metadata.num_rows == expected
    temp.replace(dest)
    return dest


def verify(source, views, templates):
    """Check that every published ID and binding survives the Parquet conversion."""
    catalog = pq.read_table(templates).to_pylist()
    lookup = {row["template_id"]: row for row in catalog}
    assert len(lookup) == len(catalog) == 12_000
    for packed, local in zip(catalog, rows(source, "templates"), strict=True):
        expected = template_record(local)
        assert packed == {
            name: expected.get(name) for name in SCHEMAS["templates"].names
        }
    families = Counter(row["base_id"] for row in catalog if row["variant_id"] == "v0")
    counts = Counter()
    view_ids = set()
    groups = defaultdict(list)
    text_pairs = defaultdict(list)
    setups = defaultdict(set)
    encoded = (
        row
        for batch in pq.ParquetFile(views).iter_batches()
        for row in batch.to_pylist()
    )
    for packed, local in zip(encoded, rows(source, "views"), strict=True):
        expected = view_record(local)
        assert packed == {name: expected[name] for name in SCHEMAS["views"].names}
        story = lookup[packed["template_id"]]
        assert story["base_id"] in families and story["variant_id"] != "v0"
        assert packed["view_id"] not in view_ids
        view_ids.add(packed["view_id"])
        counts[story["template_id"]] += 1
        base_id = packed["base_id"]
        assert base_id == story["base_id"]
        setting = packed["numeric_setting"]
        assert packed["numeric_group_id"] == f"{base_id}/n{setting}"
        text = packed["text_setting"]
        assert packed["view_id"] == f"{story['template_id']}/t{text}/n{setting}"
        operands = tuple(
            (s["step_id"], tuple(s["operands"]), s["result"])
            for s in packed["solution_steps"]
        )
        assert [step[0] for step in operands] == story["solution_step_ids"]
        binding = tuple(
            sorted((v["slot_id"], v["value"]) for v in packed["numeric_inputs"])
        )
        groups[base_id, setting].append((story["variant_id"], text, binding, operands))
        text_pairs[story["template_id"], text].append(
            (setting, packed["answer_value"], packed["text_inputs"])
        )
        setups[base_id].add(binding)
    assert len(families) == 2_000 and len(view_ids) == 100_000
    assert len(counts) == 10_000 and set(counts.values()) == {10}
    assert len(setups) == 2_000 and {len(values) for values in setups.values()} == {5}
    assert len(groups) == 10_000 and len(text_pairs) == 50_000
    for (_, setting), group in groups.items():
        expected = {
            (f"v{story}", text)
            for story in range(1, 6)
            for text in (setting, (setting - 2) % 5 + 1)
        }
        assert (
            len(group) == 10
            and {(story, text) for story, text, _, _ in group} == expected
        )
        assert len({(binding, operands) for _, _, binding, operands in group}) == 1
    for (_, text), pair in text_pairs.items():
        assert len(pair) == 2 and {entry[0] for entry in pair} == {text, text % 5 + 1}
        assert pair[0][1] != pair[1][1] and pair[0][2] == pair[1][2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    profile = json.loads((args.source / "profile.json").read_text(encoding="utf-8"))
    for name, digest in profile["files_sha256"].items():
        with (args.source / name).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == digest
    views = package(args.source, args.output, "views", view_record, profile["views"])
    templates = package(
        args.source,
        args.output,
        "templates",
        template_record,
        profile["catalog_records"],
    )
    verify(args.source, views, templates)
    print("Verified 100,000 views and 12,000 template records in Parquet.")


if __name__ == "__main__":
    main()
