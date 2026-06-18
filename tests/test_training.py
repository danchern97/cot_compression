from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from research_project_template.training import evaluate, generate, train


def tiny_cfg(tmp_path: Path):
    corpus = tmp_path / "input.txt"
    corpus.write_text(("First Citizen:\nSpeak, speak.\n" * 20), encoding="utf-8")
    run_dir = tmp_path / "run"
    return OmegaConf.create(
        {
            "paths": {"run_dir": str(run_dir)},
            "data": {
                "name": "tiny_test",
                "source_path": str(corpus),
                "prepared_dir": str(tmp_path / "prepared"),
                "download_url": None,
                "allow_download": False,
                "train_split": 0.8,
                "block_size": 8,
                "batch_size": 2,
                "eval_batch_size": 2,
            },
            "method": {
                "family": "char_gpt",
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 16,
                "dropout": 0.0,
                "bias": True,
            },
            "training": {
                "seed": 7,
                "device": "cpu",
                "max_steps": 2,
                "eval_interval": 1,
                "eval_iters": 1,
                "checkpoint_interval": 2,
                "log_interval": 1,
                "gradient_clip": 1.0,
                "compile": False,
                "deterministic": False,
            },
            "optim": {
                "lr": 0.001,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
            },
            "logging": {
                "enabled": False,
                "mode": "offline",
                "project": "tests",
                "entity": None,
                "group": None,
                "name": None,
                "tags": [],
                "log_file_name": "train.log",
            },
        }
    )


def test_training_smoke_checkpoint_and_logs(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    checkpoint = train(cfg)

    assert checkpoint.exists()
    assert (tmp_path / "run" / "resolved_config.yaml").exists()
    assert (tmp_path / "run" / "train.log").exists()


def test_evaluate_and_generate_from_checkpoint(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    checkpoint = train(cfg)

    eval_cfg = cfg.copy()
    eval_cfg.checkpoint = {"path": str(checkpoint)}
    eval_cfg.eval_iters = 1
    losses = evaluate(eval_cfg)
    assert losses["val"] > 0

    gen_cfg = OmegaConf.create(
        {
            "paths": {"run_dir": str(tmp_path / "generate")},
            "checkpoint": {"path": str(checkpoint)},
            "prompt": "First",
            "max_new_tokens": 4,
            "temperature": 1.0,
            "top_k": None,
            "output_file": "generated.txt",
            "device": "cpu",
        }
    )
    text = generate(gen_cfg)
    assert text.startswith("First")
    assert (tmp_path / "generate" / "generated.txt").exists()
