"""Render five shared numeric setups for every GSM8K story family."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from shutil import copyfile

from cot_compression.data.gsm8k.io import read_jsonl, write_jsonl
from cot_compression.data.gsm8k.sampling import numeric_cases
from cot_compression.data.gsm8k.shared_views import (
    SEED,
    TEXT_SETTINGS,
    numeric_cycle,
    render_family,
)
from cot_compression.data.gsm8k.templates import (
    render,
    solution_steps,
    step_dependencies,
)

SOURCE = Path("data/gsm8k-luna-100k-views-per-text")
OUTPUT = Path("data/gsm8k-luna-100k-shared-views")
BASE = Path("data/gsm8k-luna-6000-expanded-text-slots")
STORIES = Path("data/gsm8k-luna-6000-five-stories-partial")
VARIANT_IDS = tuple(f"v{i}" for i in TEXT_SETTINGS)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalize_question(text):
    return " ".join(text.casefold().split())


def template_record(row, variant_id):
    template = row["template"]
    steps = template["steps"]
    guarded = {arg for rule in template["constraints"] for arg in rule["args"]}
    guards = step_dependencies(steps, guarded)
    visible = solution_steps(steps, template["answer"], guards)
    base_id = row["source_id"]
    return {
        **row,
        "base_id": base_id,
        "variant_id": variant_id,
        "template_id": f"{base_id}/{variant_id}",
        "solution_step_ids": [step["id"] for step in steps if step["id"] in visible],
    }


def text_bindings(template, views):
    """Recover the text values for reserve stories from their archived renders."""
    pairs = {
        tuple((views[t, n]["question"], views[t, n]["answer"]) for n in (1, 2)): t
        for t in TEXT_SETTINGS
    }
    assert len(pairs) == len(TEXT_SETTINGS)
    cases = numeric_cases(template)
    pools = {slot["id"]: slot["values"] for slot in template["choices"]}
    bindings = {}
    for options in product(*pools.values()):
        text = dict(zip(pools, options, strict=True))
        rendered = [render(template, case, text) for case in cases]
        pair = tuple((row["question"], row["answer"]) for row in rendered)
        setting = pairs.pop(pair, None)
        if setting is not None:
            bindings[setting] = text
            if not pairs:
                return bindings
    raise ValueError("An archived question and solution pair could not be reproduced")


def reserve_families(existing, required_steps):
    bases = {
        row["source_id"]: template_record(row, "v0")
        for row in read_jsonl(BASE / "templates.jsonl")
    }
    excluded = {
        row["source_id"]
        for row in read_jsonl(BASE / "text-slot-review.jsonl")
        if row["status"] == "math_review"
    }
    stories = defaultdict(dict)
    for row in read_jsonl(STORIES / "templates.jsonl"):
        sid = row["source_id"]
        if (
            sid not in existing
            and sid not in excluded
            and len(bases[sid]["solution_step_ids"]) in required_steps
        ):
            stories[sid][row["variant_id"]] = template_record(row, row["variant_id"])
    views = defaultdict(dict)
    with (STORIES / "views.jsonl").open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            if len(stories.get(row["source_id"], {})) == 5:
                views[row["source_id"], row["variant_id"]][
                    row["text_setting"], row["numeric_setting"]
                ] = row
    for sid, variants in stories.items():
        if len(variants) == 5:
            family = [variants[vid] for vid in VARIANT_IDS]
            archived = [views[sid, vid] for vid in VARIANT_IDS]
            texts = [
                text_bindings(story["template"], rows)
                for story, rows in zip(family, archived, strict=True)
            ]
            yield (
                bases[sid],
                family,
                texts,
                [r for rows in archived for r in rows.values()],
            )


def regenerate(source, output):
    profile = json.loads((source / "profile.json").read_text(encoding="utf-8"))
    for name, digest in profile["files_sha256"].items():
        assert sha256(source / name) == digest, name
    families = defaultdict(dict)
    for row in read_jsonl(source / "templates.jsonl"):
        families[row["base_id"]][row["variant_id"]] = row
    archived = defaultdict(list)
    for row in read_jsonl(source / "views.jsonl"):
        archived[row["base_id"]].append(row)

    accepted = {}
    rejected = defaultdict(list)
    questions = set()

    def accept(base, stories, texts, previous):
        cycle = numeric_cycle(base, stories, texts, previous)
        if cycle is None:
            return False
        views = list(render_family(base, stories, texts, cycle))
        keys = {normalize_question(row["question"]) for row in views}
        if len(keys) != 50 or keys & questions:
            return False
        questions.update(keys)
        accepted[base["base_id"]] = ([base, *stories], views)
        return True

    for i, (sid, family) in enumerate(families.items(), 1):
        base = family["v0"]
        stories = [family[vid] for vid in VARIANT_IDS]
        previous = archived.pop(sid)
        texts = [
            {
                row["text_setting"]: row["text_inputs"]
                for row in previous
                if row["variant_id"] == vid
            }
            for vid in VARIANT_IDS
        ]
        assert len(previous) == 50 and all(
            set(text) == set(TEXT_SETTINGS) for text in texts
        )
        if not accept(base, stories, texts, previous):
            rejected[len(base["solution_step_ids"])].append(sid)
        if i % 500 == 0:
            print(f"Checked {i:,} families, retained {len(accepted):,}.", flush=True)

    replaced = {}
    if rejected:
        for path, digest in profile["input_sha256"].items():
            assert sha256(Path(path)) == digest, path
        for base, stories, texts, previous in reserve_families(families, rejected):
            length = len(base["solution_step_ids"])
            if rejected[length] and accept(base, stories, texts, previous):
                replaced[rejected[length].pop()] = base["base_id"]
                if not any(rejected.values()):
                    break
    # Keep the exact dataset size even when a step-count stratum has no reserve.
    for length, unresolved in rejected.items():
        if not unresolved:
            continue
        for offset in range(1, 12):
            nearby = {length - offset, length + offset}
            for base, stories, texts, previous in reserve_families(
                families.keys() | accepted.keys(), nearby
            ):
                if accept(base, stories, texts, previous):
                    replaced[unresolved.pop()] = base["base_id"]
                    if not unresolved:
                        break
            if not unresolved:
                break
    assert not any(rejected.values()), dict(rejected)
    assert len(accepted) == 2_000 and len(questions) == 100_000
    records = [accepted[replaced.get(sid, sid)] for sid in families]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "views.jsonl", (row for _, views in records for row in views))
    if replaced:
        write_jsonl(
            output / "templates.jsonl",
            (row for catalog, _ in records for row in catalog),
        )
    else:
        copyfile(source / "templates.jsonl", output / "templates.jsonl")
    profile.update(
        {
            "selection_method": (
                "Retain the prior stratified sample where five shared numeric setups are possible. "
                "Replace remaining families with verified reserves nearest in step count."
            ),
            "solution_step_distribution": dict(
                sorted(
                    Counter(
                        len(catalog[0]["solution_step_ids"]) for catalog, _ in records
                    ).items()
                )
            ),
            "stories_with_adapted_numeric_grids": sum(
                bool(story["checks"]["numeric_grids_changed"])
                for catalog, _ in records
                for story in catalog[1:]
            ),
            "numeric_sampling": {
                "seed": SEED,
                "cycle": [[t, t % 5 + 1] for t in TEXT_SETTINGS],
                "numeric_groups": 10_000,
                "numeric_setups_per_base": 5,
                "numeric_setups_per_story": 5,
                "replaced_families": replaced,
            },
            "ids": {
                **profile["ids"],
                "numeric_group_id": "{base_id}/n{numeric_setting}",
                "view_id": "{template_id}/t{text_setting}/n{numeric_setting}",
            },
            "checks": {
                "replayed_story_graphs": 10_000,
                "replayed_question_and_solution_pairs": 100_000,
                "normalized_duplicate_questions": 0,
                "text_settings_with_two_distinct_answers": 50_000,
            },
            "resampled_from": str(source),
            "files_sha256": {
                name: sha256(output / name)
                for name in ("views.jsonl", "templates.jsonl")
            },
        }
    )
    (output / "profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Rendered 100,000 views. Replaced {len(replaced)} families. Output: {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Choose an output directory separate from the source snapshot")
    regenerate(args.source, args.output)


if __name__ == "__main__":
    main()
