"""Per-iteration + final report writer."""

from __future__ import annotations

import difflib
import html
import json
from pathlib import Path

from .types import Edit, IterResult, RowScore


class Reporter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ----- baseline -----
    def write_baseline(
        self, skill: str, val_scores: list[RowScore], test_scores: list[RowScore]
    ) -> None:
        d = self.run_dir / "baseline"
        d.mkdir(exist_ok=True)
        (d / "skill.md").write_text(skill)
        (d / "val_scores.json").write_text(
            json.dumps([s.to_dict() for s in val_scores], indent=2)
        )
        (d / "test_scores.json").write_text(
            json.dumps([s.to_dict() for s in test_scores], indent=2)
        )

    def write_train(self, scores: list[RowScore], *, iter_num: int) -> None:
        d = self.run_dir / "train"
        d.mkdir(exist_ok=True)
        (d / f"train_scores_iter{iter_num:03d}.json").write_text(
            json.dumps([s.to_dict() for s in scores], indent=2)
        )

    # ----- per-iter -----
    def write_iter(
        self,
        iter_num: int,
        old_skill: str,
        new_skill: str,
        edit: Edit,
        val_scores: list[RowScore],
        ir: IterResult,
    ) -> None:
        d = self.run_dir / f"iter-{iter_num:03d}"
        d.mkdir(exist_ok=True)
        (d / "skill.md").write_text(new_skill)
        (d / "edit.json").write_text(json.dumps(edit.to_dict(), indent=2))
        (d / "status.json").write_text(json.dumps(ir.to_dict(), indent=2))
        if val_scores:
            (d / "val_scores.json").write_text(
                json.dumps([s.to_dict() for s in val_scores], indent=2)
            )
        diff = difflib.unified_diff(
            old_skill.splitlines(keepends=True),
            new_skill.splitlines(keepends=True),
            fromfile=f"iter-{iter_num-1:03d}/skill.md",
            tofile=f"iter-{iter_num:03d}/skill.md",
        )
        (d / "diff.patch").write_text("".join(diff))
        (d / "rationale.md").write_text(
            f"# Iter {iter_num} — {'ACCEPTED' if ir.accepted else 'REJECTED'}\n\n"
            f"**Edit op:** `{edit.op}`  \n"
            f"**Locator:** `{edit.locator}`  \n"
            f"**Val score:** {ir.val_score_before:.3f} -> {ir.val_score_after:.3f}\n\n"
            f"**Rationale (from optimizer):** {edit.rationale}\n\n"
            f"**Why rejected:** {ir.reason or '(accepted)'}\n"
        )

    # ----- final -----
    def write_final(
        self, final_skill: str, final_test_scores: list[RowScore], summary: dict
    ) -> None:
        (self.run_dir / "final_skill.md").write_text(final_skill)
        (self.run_dir / "final_test_scores.json").write_text(
            json.dumps([s.to_dict() for s in final_test_scores], indent=2)
        )
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (self.run_dir / "summary.md").write_text(_summary_md(summary))
        (self.run_dir / "report.html").write_text(_html_report(summary, self.run_dir))


def _summary_md(summary: dict) -> str:
    lines = [
        f"# SkillOpt run — {Path(summary['skill_path']).name}",
        "",
        f"- baseline_val: **{summary['baseline_val']:.3f}**",
        f"- baseline_test: **{summary['baseline_test']:.3f}**",
        f"- final_val: **{summary['final_val']:.3f}**",
        f"- final_test: **{summary['final_test']:.3f}**",
        f"- accepted edits: {summary['accepted_edits']} / {summary['iters']}",
        f"- rejected edits: {summary['rejected_edits']} / {summary['iters']}",
        "",
        "## Iter log",
        "",
        "| iter | op | val_before | val_after | Δ | status | lr | edit_chars | rationale |",
        "|------|----|-----------:|----------:|----:|--------|---:|-----------:|-----------|",
    ]
    for ir in summary["iters_log"]:
        delta = ir["val_score_after"] - ir["val_score_before"]
        status = "✓ accept" if ir["accepted"] else "✗ reject"
        rat = (ir["edit"]["rationale"] or "")[:80].replace("|", "\\|")
        lines.append(
            f"| {ir['iter_num']} | {ir['edit']['op']} | "
            f"{ir['val_score_before']:.3f} | {ir['val_score_after']:.3f} | "
            f"{delta:+.3f} | {status} | "
            f"{ir.get('lr_budget_chars', 0)} | {ir.get('edit_size_chars', 0)} | {rat} |"
        )
    return "\n".join(lines) + "\n"


def _html_report(summary: dict, run_dir: Path) -> str:
    # Build score curve as inline SVG
    pts_x = list(range(len(summary["iters_log"]) + 1))
    pts_y = [summary["baseline_val"]] + [ir["val_score_after"] for ir in summary["iters_log"]]
    # Track the "current accepted" curve too
    cur_y = [summary["baseline_val"]]
    running = summary["baseline_val"]
    for ir in summary["iters_log"]:
        if ir["accepted"]:
            running = ir["val_score_after"]
        cur_y.append(running)

    svg = _svg_chart(pts_x, pts_y, cur_y)

    rows_html = []
    for ir in summary["iters_log"]:
        n = ir["iter_num"]
        diff_path = run_dir / f"iter-{n:03d}" / "diff.patch"
        diff_text = diff_path.read_text() if diff_path.exists() else ""
        accept_badge = (
            '<span style="background:#1f8a3e;color:#fff;padding:2px 8px;border-radius:4px">ACCEPTED</span>'
            if ir["accepted"]
            else '<span style="background:#a33;color:#fff;padding:2px 8px;border-radius:4px">REJECTED</span>'
        )
        delta = ir["val_score_after"] - ir["val_score_before"]
        rows_html.append(
            f"""
        <details style="margin-bottom:8px; border:1px solid #ddd; border-radius:6px; padding:8px">
          <summary>
            <b>iter {n}</b> &nbsp; {accept_badge} &nbsp;
            <code>{html.escape(ir['edit']['op'])}</code> &nbsp;
            val {ir['val_score_before']:.3f} → {ir['val_score_after']:.3f}
            ({delta:+.3f}) &nbsp;
            <span style="color:#888">lr={ir.get('lr_budget_chars', 0)}ch
            (edit={ir.get('edit_size_chars', 0)}ch)</span> &nbsp;
            <i>{html.escape(ir['edit']['rationale'][:120])}</i>
          </summary>
          <p><b>Locator:</b> <code>{html.escape(ir['edit']['locator'][:200])}</code></p>
          <p><b>Reason:</b> {html.escape(ir['reason'] or '(accepted)')}</p>
          <pre style="background:#f6f8fa;padding:8px;overflow:auto;max-height:400px">{html.escape(diff_text)}</pre>
        </details>
        """
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SkillOpt — {html.escape(Path(summary['skill_path']).name)}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1, h2 {{ border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  .kpi {{ display: inline-block; padding: 12px 20px; border: 1px solid #ddd; border-radius: 8px; margin-right: 8px; }}
  .kpi b {{ font-size: 1.6em; display:block }}
  code {{ background:#f6f8fa; padding:2px 4px; border-radius:3px }}
</style></head><body>
<h1>SkillOpt — {html.escape(Path(summary['skill_path']).name)}</h1>
<p>
  <span class="kpi">baseline test<br><b>{summary['baseline_test']:.3f}</b></span>
  <span class="kpi">final test<br><b>{summary['final_test']:.3f}</b></span>
  <span class="kpi">Δ test<br><b>{summary['final_test'] - summary['baseline_test']:+.3f}</b></span>
  <span class="kpi">accepted / total<br><b>{summary['accepted_edits']} / {summary['iters']}</b></span>
</p>
<h2>Validation score over iterations</h2>
{svg}
<p><small>Blue dots = score after applying that iter's proposed edit (whether accepted or not).
Green line = current accepted-skill score (only steps up on accept).</small></p>
<h2>Iterations</h2>
{''.join(rows_html)}
</body></html>"""


def _svg_chart(xs: list[int], ys: list[float], cur_ys: list[float]) -> str:
    if not xs:
        return ""
    W, H, P = 800, 260, 36
    xmin, xmax = min(xs), max(xs) or 1
    ymin = min(min(ys), min(cur_ys), 0.0)
    ymax = max(max(ys), max(cur_ys), 1.0)
    if ymax == ymin:
        ymax = ymin + 1

    def sx(x: float) -> float:
        return P + (x - xmin) / (xmax - xmin) * (W - 2 * P)

    def sy(y: float) -> float:
        return H - P - (y - ymin) / (ymax - ymin) * (H - 2 * P)

    line_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    cur_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, cur_ys))
    dots = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#1f6feb"/>'
        for x, y in zip(xs, ys)
    )
    axis = (
        f'<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#888"/>'
        f'<line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#888"/>'
        f'<text x="{P}" y="{H-P+18}" font-size="11" fill="#666">iter {xmin}</text>'
        f'<text x="{W-P}" y="{H-P+18}" font-size="11" fill="#666" text-anchor="end">iter {xmax}</text>'
        f'<text x="{P-6}" y="{P}" font-size="11" fill="#666" text-anchor="end">{ymax:.2f}</text>'
        f'<text x="{P-6}" y="{H-P}" font-size="11" fill="#666" text-anchor="end">{ymin:.2f}</text>'
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}">'
        f"{axis}"
        f'<polyline fill="none" stroke="#1f6feb" stroke-width="1.5" stroke-dasharray="3,3" points="{line_pts}"/>'
        f'<polyline fill="none" stroke="#1f8a3e" stroke-width="2.5" points="{cur_pts}"/>'
        f"{dots}"
        f"</svg>"
    )
