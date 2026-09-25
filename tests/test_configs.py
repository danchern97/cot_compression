from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_hydra_configs_compose() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        sft_cfg = compose(
            config_name="run",
            overrides=["logging.enabled=false"],
        )
        eval_cfg = compose(
            config_name="run",
            overrides=["workflow=evaluate_methods", "logging.enabled=false"],
        )
        eval_32b_cfg = compose(
            config_name="run",
            overrides=[
                "workflow=evaluate_methods",
                "data=dolci_think_sft_32b_600k",
                "logging.enabled=false",
            ],
        )
        sft_06b_cfg = compose(
            config_name="run",
            overrides=["workflow=sft_train_06b", "logging.enabled=false"],
        )

    assert sft_cfg.mode == "sft_train"
    assert sft_cfg.method.model_name == "Qwen/Qwen3-4B"
    assert sft_cfg.data.source_name == "allenai/Dolci-Think-SFT-7B"
    assert sft_cfg.data.name == "dolci_think_sft_7b_600k"
    assert sft_cfg.data.eval_size == 20000
    assert sft_cfg.data.test_size == 100000
    # fp32 master weights: bf16 params would put the AdamW moments in bf16 too and
    # updates at lr 1e-5 would round away.
    assert sft_cfg.training.torch_dtype == "float32"
    assert sft_cfg.training.autocast_dtype == "bfloat16"
    assert sft_cfg.training.attn_implementation == "sdpa"
    # Rows are dropped, never truncated, at the full model context.
    assert sft_cfg.training.max_length == 32768
    assert sft_cfg.training.max_batch_tokens >= sft_cfg.training.max_length
    assert sft_cfg.training.target_global_batch == 32
    assert sft_cfg.training.resume_from_checkpoint == "auto"
    assert sft_cfg.training.max_steps is None
    # Heavy artifacts on scratch, light ones on HOME (200 GiB quota).
    assert sft_cfg.paths.output_dir.startswith("/scratch-shared/")
    assert sft_cfg.paths.tokenized_dir.startswith("/scratch-shared/")
    assert sft_cfg.paths.data_dir.startswith("/scratch-shared/")
    # W&B: one project, and artifact uploads off by default (tokens.jsonl is GBs).
    assert sft_cfg.logging.entity == "cot-compression"
    assert sft_cfg.logging.project == "cot-compression-qwen3"
    assert sft_cfg.logging.log_artifacts is False
    assert eval_cfg.mode == "evaluate_methods"
    assert eval_cfg.method.model_name == "Qwen/Qwen3-4B"
    assert eval_cfg.evaluation.max_length is None
    assert eval_cfg.evaluation.batch_size == 64
    assert eval_cfg.evaluation.num_workers == 14
    assert eval_cfg.evaluation.prefetch_factor == 4
    assert eval_cfg.evaluation.methods.patching.compression_ratio == 2.0
    assert eval_cfg.evaluation.methods.entropy_weighted_mean.temperature == 1.0
    assert eval_cfg.evaluation.methods.surprisal_weighted_mean.temperature == 1.0
    assert list(eval_cfg.evaluation.precompute_signals) == ["entropy", "surprisal"]
    assert eval_cfg.evaluation.entropy_cache_dir is None
    assert eval_cfg.evaluation.scratch_dir is None
    assert eval_32b_cfg.data.source_name == "allenai/Dolci-Think-SFT-32B"
    assert eval_32b_cfg.data.name == "dolci_think_sft_32b_600k"

    # The 0.6B campaign: same dispatch and dataset, different model.
    assert sft_06b_cfg.mode == "sft_train"
    assert sft_06b_cfg.method.model_name == "Qwen/Qwen3-0.6B"
    assert sft_06b_cfg.data.name == sft_cfg.data.name
    # Checkpointing stays ON. The probe measured 82.8-88.0 GiB (89-94% of a 93 GiB
    # card) without it at the real 32,768-token micro-batch, against 20.5 with.
    # Pinned so the ~30% it costs is never quietly "optimized" away again.
    assert sft_06b_cfg.training.gradient_checkpointing is True
    assert sft_06b_cfg.training.ce_chunk_tokens == 4096
    assert sft_06b_cfg.training.checkpoint_minutes == 45
    # Inherited from sft_full_bf16 and load-bearing: fp32 masters, no truncation.
    assert sft_06b_cfg.training.torch_dtype == "float32"
    assert sft_06b_cfg.training.autocast_dtype == "bfloat16"
    assert sft_06b_cfg.training.resume_from_checkpoint == "auto"
    # Everything plan_signature covers must match the 4B run, so the two walk the
    # identical curriculum and their eval losses are comparable.
    for knob in (
        "seed",
        "max_length",
        "max_batch_tokens",
        "micro_batch_max_sequences",
        "length_group_size",
        "eval_examples",
    ):
        assert sft_06b_cfg.training[knob] == sft_cfg.training[knob], knob
    # run_name is the run directory AND the W&B run id, and carries no model
    # identifier -- so the 0.6B campaign must not default to the 4B's run_tag.
    assert sft_06b_cfg.run_tag == "qwen3-0.6b"
    assert sft_06b_cfg.run_tag != sft_cfg.run_tag
    assert sft_06b_cfg.paths.run_dir != sft_cfg.paths.run_dir


def test_encoder_workflow_composes() -> None:
    """The encoder campaign's defaults, and that the grid is reachable by override.

    Pinned because two of these are silent failure modes rather than errors: a
    shared `run_tag` would resolve the encoder run onto an SFT run's directory,
    and truncating instead of dropping over-length rows would cut placeholder
    slots off and desynchronize `slot_positions` from K.
    """
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="run",
            overrides=["workflow=encoder_train", "logging.enabled=false"],
        )
        sft_cfg = compose(config_name="run", overrides=["logging.enabled=false"])

        assert cfg.mode == "encoder_train"
        assert cfg.data.name == "dolci_compression_traces"
        assert cfg.method.model_name == "Qwen/Qwen3-0.6B"

        # The settled baseline. Each of these is a deliberate choice documented in
        # ENCODER_PLAN.md, not a value awaiting tuning.
        assert cfg.encoder.n_blocks == 2
        assert cfg.encoder.codebook_size == 64
        assert cfg.encoder.latent_init == "random"
        assert cfg.encoder.patching == "uniform"
        assert cfg.encoder.memory_layer == -1
        assert cfg.evaluation.methods.patching.compression_ratio == 4.0

        # fp32 encoder masters (bf16 moments round updates away at these LRs);
        # bf16 frozen decoder, which carries no optimizer state.
        assert cfg.training.torch_dtype == "float32"
        assert cfg.training.decoder_dtype == "bfloat16"

        # Aux-pass compilation, sized from the census over this exact plan: an
        # ADDITIVE width ladder (a power-of-two one doubled W and OOM'd) and a
        # recompile limit that is fatal rather than a silent eager fallback.
        assert cfg.training.aux_width_multiple == 1024
        # Reported beside base/no_cot at every eval; renaming one renames its W&B key.
        assert list(cfg.encoder.eval_controls) == ["random", "surprisal_t0"]
        assert cfg.training.aux_recompile_limit == 128

        # Positions. Not an ablation: without them a slot can only find its own
        # span in the memory by content match. `run_name` carries the choice, so a
        # rope arm and a none arm cannot resolve onto one directory.
        assert cfg.encoder.position_encoding == "rope"
        assert cfg.encoder.rope_theta == 1000000.0
        assert "_perope_" in cfg.run_name
        # The next-patch weight changes what is trained, so it must separate run
        # directories too -- or an arm at another lambda with the same run_tag would
        # resume this one's checkpoint.
        assert "_np1.0_" in cfg.run_name

        # Batch ORDER: global length sort, rows shuffled inside 1024-row buckets,
        # and whole micro-steps shuffled -- the length-grouped default marched every
        # step from the longest rows to the shortest. The bigger global batch is
        # what lets one step span several length ranges (and so several domains).
        assert cfg.training.length_bucket_rows == 1024
        assert cfg.training.shuffle_micro_steps is True
        assert cfg.training.target_global_batch == 128
        # The SFT campaigns keep the batches they trained on, so their plan
        # signature is unchanged and their checkpoints still resume.
        assert sft_cfg.training.length_bucket_rows is None
        assert sft_cfg.training.shuffle_micro_steps is False
        assert sft_cfg.training.target_global_batch == 32

        # run_name carries no model identifier, so the encoder run must not share
        # a run_tag -- or resume_from_checkpoint=auto loads the wrong weights.
        assert cfg.run_tag != sft_cfg.run_tag
        assert cfg.paths.run_dir != sft_cfg.paths.run_dir

        # All eight grid arms must be reachable by override alone, with distinct
        # run directories so they cannot overwrite one another.
        names = set()
        for init in ("random", "surprisal_t0"):
            for mask in ("causal", "bidirectional"):
                for codes in (64, 1024):
                    arm = compose(
                        config_name="run",
                        overrides=[
                            "workflow=encoder_train",
                            "logging.enabled=false",
                            f"encoder.latent_init={init}",
                            f"encoder.self_attn_mask={mask}",
                            f"encoder.cross_attn_mask={mask}",
                            f"encoder.codebook_size={codes}",
                        ],
                    )
                    names.add(arm.run_name)
        assert len(names) == 8, names


def test_eval_split_selection_is_corpus_agnostic() -> None:
    """`evaluation.split` is semantic, so neither corpus's naming leaks into config.

    Phase 1 scores `test`; in-training eval scores `validation`. Getting this wrong
    would report a number the model selected checkpoints on, which is not held out.
    """
    from unittest.mock import patch

    from cot_compression.training.evaluate import load_eval_dataset

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        traces = compose(
            config_name="run",
            overrides=[
                "workflow=evaluate_methods",
                "data=dolci_compression_traces",
                "logging.enabled=false",
            ],
        )
        sft = compose(
            config_name="run",
            overrides=["workflow=evaluate_methods", "logging.enabled=false"],
        )

    assert traces.data.loader == "traces"
    assert sft.data.loader == "dolci_sft"
    assert traces.evaluation.split is None  # => validation

    class _Splits:
        train, val, eval, test = "TRAIN", "VAL", "EVAL", "TEST"

    with patch(
        "cot_compression.training.evaluate.load_trace_data", return_value=_Splits()
    ):
        assert load_eval_dataset(traces) == "VAL"
        traces.evaluation.split = "test"
        assert load_eval_dataset(traces) == "TEST"

    with patch(
        "cot_compression.training.evaluate.load_dolci_sft_data", return_value=_Splits()
    ):
        assert load_eval_dataset(sft) == "EVAL"
        sft.evaluation.split = "test"
        assert load_eval_dataset(sft) == "TEST"
        sft.evaluation.split = "nonsense"
        try:
            load_eval_dataset(sft)
            raise AssertionError("expected an unknown-split error")
        except ValueError as error:
            assert "Unknown evaluation.split" in str(error)
