"""The SkillOpt training loop."""

from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from .optimizer import apply_edit, propose_edit
from .reporter import Reporter
from .rollout import rollout
from .scorers import Scorer
from .types import Edit, EvalRow, IterResult, RowScore


@dataclass
class LoopConfig:
    iters: int = 15
    train_frac: float = 0.6
    val_frac: float = 0.2
    # test_frac is the remainder
    seed: int = 42
    rollout_model: str = ""
    optimizer_model: str = ""
    parallel_rollouts: int = 6
    max_rejected_in_context: int = 8
    epsilon: float = 1e-6  # strict-improvement tolerance
    # Drop rows whose baseline score is below this — filters out
    # generator-mis-specified rows where the skill is correct but the expected
    # fingerprint is wrong. 0.0 disables the filter.
    drop_baseline_below: float = 0.0
    # Textual learning-rate budget (per-iter char limit on `new_text`).
    # Initial budget shrinks by `lr_decay` after every rejected edit and grows by
    # `lr_growth` after every accepted edit, clamped to [floor, ceiling].
    # This is the SkillOpt paper's primary stabilizer — when the optimizer gets
    # stuck rejecting on the same locator, the shrinking budget forces a smaller,
    # more surgical edit which usually breaks the rut.
    lr_chars_initial: int = 400
    lr_chars_floor: int = 80
    lr_chars_ceiling: int = 800
    lr_decay: float = 0.7
    lr_growth: float = 1.3
    # Locator cooldown: after a locator has been rejected this many times within
    # the rolling window, refuse further edits on it (count as auto-rejected).
    # This is the diversity stabilizer that the LR budget cannot provide alone.
    locator_reject_threshold: int = 2
    locator_cooldown_window: int = 5


def _split_rows(
    rows: list[EvalRow], cfg: LoopConfig
) -> tuple[list[EvalRow], list[EvalRow], list[EvalRow]]:
    rng = random.Random(cfg.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def _score_set(
    skill: str,
    rows: list[EvalRow],
    scorer: Scorer,
    cfg: LoopConfig,
    config: Config,
) -> list[RowScore]:
    """Roll out + score every row; parallelized."""
    results: list[RowScore] = [None] * len(rows)  # type: ignore[list-item]

    def _one(i: int, row: EvalRow) -> tuple[int, RowScore]:
        out = rollout(skill, row, cfg.rollout_model or config.test_model, config)
        sc = scorer.score(row, out)
        return i, sc

    with ThreadPoolExecutor(max_workers=cfg.parallel_rollouts) as ex:
        futures = [ex.submit(_one, i, row) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            i, sc = fut.result()
            results[i] = sc
    return results


def _mean(scores: list[RowScore]) -> float:
    if not scores:
        return 0.0
    return round(sum(s.score for s in scores) / len(scores), 4)


class SkillOptLoop:
    def __init__(
        self,
        *,
        skill_path: Path,
        rows: list[EvalRow],
        scorer: Scorer,
        config: Config,
        loop_config: LoopConfig | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.skill_path = Path(skill_path)
        self.original_skill = self.skill_path.read_text()
        self.rows = rows
        self.scorer = scorer
        self.config = config
        self.cfg = loop_config or LoopConfig()
        run_id = (
            f"{self.skill_path.stem}-"
            f"{time.strftime('%Y%m%d-%H%M%S')}-"
            f"{hashlib.sha1(self.original_skill.encode()).hexdigest()[:6]}"
        )
        self.run_dir = run_dir or Path("results/skillopt") / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.reporter = Reporter(self.run_dir)

    def run(self) -> dict:
        if self.cfg.drop_baseline_below > 0.0:
            print(
                f"[skillopt] baseline-filter pass on all {len(self.rows)} rows "
                f"(drop if score < {self.cfg.drop_baseline_below})..."
            )
            baseline_filter_scores = _score_set(
                self.original_skill, self.rows, self.scorer, self.cfg, self.config
            )
            keep_ids = {
                s.row_id for s in baseline_filter_scores if s.score >= self.cfg.drop_baseline_below
            }
            dropped = [s for s in baseline_filter_scores if s.row_id not in keep_ids]
            self.rows = [r for r in self.rows if r.id in keep_ids]
            (self.run_dir / "dropped_rows.json").write_text(
                json.dumps([s.to_dict() for s in dropped], indent=2)
            )
            print(
                f"[skillopt] kept {len(self.rows)}, dropped {len(dropped)} "
                f"(see dropped_rows.json)"
            )

        train, val, test = _split_rows(self.rows, self.cfg)
        print(
            f"[skillopt] split: train={len(train)} val={len(val)} test={len(test)} "
            f"(seed={self.cfg.seed})"
        )

        # Persist run config + splits
        (self.run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "skill_path": str(self.skill_path),
                    "iters": self.cfg.iters,
                    "train": [r.id for r in train],
                    "val": [r.id for r in val],
                    "test": [r.id for r in test],
                    "rollout_model": self.cfg.rollout_model or self.config.test_model,
                    "optimizer_model": self.cfg.optimizer_model or self.config.model,
                },
                indent=2,
            )
        )

        # Baseline
        print("[skillopt] scoring baseline on val + test...")
        current = self.original_skill
        baseline_val = _score_set(current, val, self.scorer, self.cfg, self.config)
        baseline_test = _score_set(current, test, self.scorer, self.cfg, self.config)
        baseline_val_mean = _mean(baseline_val)
        baseline_test_mean = _mean(baseline_test)
        self.reporter.write_baseline(current, baseline_val, baseline_test)
        print(
            f"[skillopt] baseline val={baseline_val_mean:.3f} test={baseline_test_mean:.3f}"
        )

        # Training rollouts (one pass — used to seed failing-row pool)
        print("[skillopt] scoring train rollouts (for failure pool)...")
        train_scores = _score_set(current, train, self.scorer, self.cfg, self.config)
        self.reporter.write_train(train_scores, iter_num=0)

        rejected: list[Edit] = []
        iters_log: list[IterResult] = []
        current_val_mean = baseline_val_mean
        lr_chars = self.cfg.lr_chars_initial
        # Locator cooldown: track rejects per locator within a sliding window
        # of iter numbers. A locator is "on cooldown" if its reject count in the
        # last `locator_cooldown_window` iters is >= `locator_reject_threshold`.
        locator_reject_iters: dict[str, list[int]] = {}

        def _cooldown_set(current_iter: int) -> set[str]:
            window_start = current_iter - self.cfg.locator_cooldown_window
            return {
                loc
                for loc, iters in locator_reject_iters.items()
                if sum(1 for i in iters if i >= window_start)
                >= self.cfg.locator_reject_threshold
            }

        for it in range(1, self.cfg.iters + 1):
            print(
                f"\n[skillopt] === iter {it}/{self.cfg.iters} "
                f"(lr_budget={lr_chars} chars) ==="
            )
            cooldown = _cooldown_set(it)
            if cooldown:
                print(f"[skillopt] cooldown locators: {sorted(cooldown)}")

            # Build failure context: pair (prompt, score) sorted worst-first
            row_by_id = {r.id: r for r in train}
            failures = sorted(
                [(row_by_id[s.row_id].prompt, s) for s in train_scores if s.score < 1.0],
                key=lambda ps: ps[1].score,
            )
            try:
                edit = propose_edit(
                    skill=current,
                    failures=failures,
                    rejected=rejected[-self.cfg.max_rejected_in_context :],
                    model=self.cfg.optimizer_model or self.config.model,
                    config=self.config,
                    lr_chars=lr_chars,
                    cooldown_locators=sorted(cooldown),
                )
            except Exception as e:
                print(f"[skillopt] optimizer error: {e}; skipping iter")
                continue

            print(
                f"[skillopt] proposed [{edit.op}] locator={edit.locator[:60]!r} "
                f"new_text_chars={len(edit.new_text)} "
                f"rationale={edit.rationale[:100]!r}"
            )

            # Enforce cooldown server-side: if optimizer ignored the ban, auto-reject
            # without scoring (saves a val pass).
            if any(loc in edit.locator or edit.locator in loc for loc in cooldown):
                print(f"[skillopt] auto-REJECT iter {it}: locator on cooldown")
                rejected.append(edit)
                locator_reject_iters.setdefault(edit.locator, []).append(it)
                lr_chars = max(
                    self.cfg.lr_chars_floor, int(lr_chars * self.cfg.lr_decay)
                )
                ir = IterResult(
                    it,
                    edit,
                    False,
                    current_val_mean,
                    current_val_mean,
                    "locator on cooldown",
                    lr_budget_chars=lr_chars,
                    edit_size_chars=len(edit.new_text),
                )
                iters_log.append(ir)
                self.reporter.write_iter(it, current, current, edit, [], ir)
                continue

            new_skill, err = apply_edit(current, edit)
            if err:
                print(f"[skillopt] apply_edit failed: {err}; treating as rejected")
                rejected.append(edit)
                # Apply-failure counts as a reject for LR-budget purposes
                lr_chars = max(
                    self.cfg.lr_chars_floor, int(lr_chars * self.cfg.lr_decay)
                )
                ir = IterResult(
                    it,
                    edit,
                    False,
                    current_val_mean,
                    current_val_mean,
                    err,
                    lr_budget_chars=lr_chars,
                    edit_size_chars=len(edit.new_text),
                )
                iters_log.append(ir)
                self.reporter.write_iter(it, current, new_skill, edit, [], ir)
                continue

            # Re-score val
            new_val_scores = _score_set(new_skill, val, self.scorer, self.cfg, self.config)
            new_val_mean = _mean(new_val_scores)
            accepted = new_val_mean > current_val_mean + self.cfg.epsilon

            # Update LR budget based on outcome
            if accepted:
                lr_chars = min(
                    self.cfg.lr_chars_ceiling, int(lr_chars * self.cfg.lr_growth)
                )
            else:
                lr_chars = max(
                    self.cfg.lr_chars_floor, int(lr_chars * self.cfg.lr_decay)
                )

            ir = IterResult(
                iter_num=it,
                edit=edit,
                accepted=accepted,
                val_score_before=current_val_mean,
                val_score_after=new_val_mean,
                reason="" if accepted else "no strict val improvement",
                lr_budget_chars=lr_chars,
                edit_size_chars=len(edit.new_text),
            )
            iters_log.append(ir)
            self.reporter.write_iter(it, current, new_skill, edit, new_val_scores, ir)

            if accepted:
                print(
                    f"[skillopt] ACCEPT iter {it}: val {current_val_mean:.3f} -> "
                    f"{new_val_mean:.3f}  (lr_budget -> {lr_chars})"
                )
                current = new_skill
                current_val_mean = new_val_mean
                # Refresh train scores since skill changed
                train_scores = _score_set(current, train, self.scorer, self.cfg, self.config)
                self.reporter.write_train(train_scores, iter_num=it)
            else:
                print(
                    f"[skillopt] REJECT iter {it}: val {current_val_mean:.3f} -> "
                    f"{new_val_mean:.3f}  (lr_budget -> {lr_chars})"
                )
                rejected.append(edit)
                locator_reject_iters.setdefault(edit.locator, []).append(it)

        # Final test scoring
        print("\n[skillopt] scoring final on test set...")
        final_test = _score_set(current, test, self.scorer, self.cfg, self.config)
        final_test_mean = _mean(final_test)
        print(f"[skillopt] FINAL: test {baseline_test_mean:.3f} -> {final_test_mean:.3f}")

        summary = {
            "skill_path": str(self.skill_path),
            "iters": self.cfg.iters,
            "baseline_val": baseline_val_mean,
            "baseline_test": baseline_test_mean,
            "final_val": current_val_mean,
            "final_test": final_test_mean,
            "accepted_edits": sum(1 for ir in iters_log if ir.accepted),
            "rejected_edits": sum(1 for ir in iters_log if not ir.accepted),
            "iters_log": [ir.to_dict() for ir in iters_log],
        }
        self.reporter.write_final(current, final_test, summary)
        return summary
