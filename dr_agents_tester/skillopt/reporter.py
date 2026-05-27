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
        edits: list[Edit],
        val_scores: list[RowScore],
        ir: IterResult,
    ) -> None:
        d = self.run_dir / f"iter-{iter_num:03d}"
        d.mkdir(exist_ok=True)
        (d / "skill.md").write_text(new_skill)
        (d / "edits.json").write_text(
            json.dumps([e.to_dict() for e in edits], indent=2)
        )
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
        edit_lines = "\n".join(
            f"- **[{e.op}]** `{e.locator}` — {e.rationale}" for e in edits
        ) or "_(no edits applied)_"
        (d / "rationale.md").write_text(
            f"# Iter {iter_num} — {'ACCEPTED' if ir.accepted else 'REJECTED'}\n\n"
            f"**L_t budget:** {ir.lt_budget} edits  \n"
            f"**Candidates proposed:** {ir.num_candidates}  \n"
            f"**Edits applied:** {ir.num_applied}  \n"
            f"**Val score:** {ir.val_score_before:.3f} -> {ir.val_score_after:.3f}"
            f"{' (cache hit)' if ir.cache_hit else ''}\n\n"
            f"**Edits:**\n{edit_lines}\n\n"
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
        "| iter | L_t | applied | val_before | val_after | Δ | status | top rationale |",
        "|------|----:|--------:|-----------:|----------:|----:|--------|---------------|",
    ]
    for ir in summary["iters_log"]:
        delta = ir["val_score_after"] - ir["val_score_before"]
        status = "✓ accept" if ir["accepted"] else "✗ reject"
        edits = ir.get("edits", [])
        rat = ((edits[0]["rationale"] if edits else ir.get("reason", "")) or "")[:70].replace("|", "\\|")
        lines.append(
            f"| {ir['iter_num']} | {ir.get('lt_budget', 0)} | {ir.get('num_applied', 0)} | "
            f"{ir['val_score_before']:.3f} | {ir['val_score_after']:.3f} | "
            f"{delta:+.3f} | {status} | {rat} |"
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
        edits = ir.get("edits", [])
        top_rat = edits[0]["rationale"] if edits else (ir.get("reason") or "")
        edits_html = "".join(
            f"<li><code>{html.escape(e['op'])}</code> "
            f"<b>{html.escape(e['locator'][:120])}</b> — {html.escape(e['rationale'][:200])}</li>"
            for e in edits
        ) or "<li><i>no edits applied</i></li>"
        cache_tag = ' <span style="color:#888">[cache hit]</span>' if ir.get("cache_hit") else ""
        rows_html.append(
            f"""
        <details style="margin-bottom:8px; border:1px solid #ddd; border-radius:6px; padding:8px">
          <summary>
            <b>iter {n}</b> &nbsp; {accept_badge} &nbsp;
            <span style="color:#888">L_t={ir.get('lt_budget', 0)}, applied {ir.get('num_applied', 0)}/{ir.get('num_candidates', 0)}</span> &nbsp;
            val {ir['val_score_before']:.3f} → {ir['val_score_after']:.3f}
            ({delta:+.3f}){cache_tag} &nbsp;
            <i>{html.escape(top_rat[:120])}</i>
          </summary>
          <p><b>Edits applied this step:</b></p>
          <ul>{edits_html}</ul>
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
    W, H, P = 800, 300, 48
    xmin, xmax = min(xs), max(xs) or 1

    # Auto-fit the y-axis to the actual data range (NOT 0..1) so small
    # iteration-to-iteration changes are visible. Pad by 8% of the data span
    # on each side; enforce a minimum visible window so a near-flat run still
    # renders with breathing room rather than a zero-height band.
    all_y = ys + cur_ys
    dmin, dmax = min(all_y), max(all_y)
    span = dmax - dmin
    min_window = 0.05  # always show at least a 5-pt window
    if span < min_window:
        mid = (dmin + dmax) / 2
        dmin, dmax = mid - min_window / 2, mid + min_window / 2
        span = dmax - dmin
    pad = span * 0.08
    ymin, ymax = dmin - pad, dmax + pad

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
    # Horizontal gridlines + y labels at 5 evenly spaced ticks across the
    # fitted range, so the reader can read off the magnitude of the moves.
    gridlines = []
    n_ticks = 5
    for i in range(n_ticks):
        yval = ymin + (ymax - ymin) * i / (n_ticks - 1)
        yy = sy(yval)
        gridlines.append(
            f'<line x1="{P}" y1="{yy:.1f}" x2="{W-P}" y2="{yy:.1f}" '
            f'stroke="#eee" stroke-width="1"/>'
            f'<text x="{P-6}" y="{yy+4:.1f}" font-size="11" fill="#666" '
            f'text-anchor="end">{yval:.3f}</text>'
        )
    # Dashed baseline reference at the first current-skill value.
    base_y = sy(cur_ys[0])
    baseline_ref = (
        f'<line x1="{P}" y1="{base_y:.1f}" x2="{W-P}" y2="{base_y:.1f}" '
        f'stroke="#bbb" stroke-width="1" stroke-dasharray="2,3"/>'
        f'<text x="{W-P}" y="{base_y-5:.1f}" font-size="10" fill="#999" '
        f'text-anchor="end">baseline {cur_ys[0]:.3f}</text>'
    )
    axis = (
        "".join(gridlines)
        + baseline_ref
        + f'<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#888"/>'
        + f'<line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#888"/>'
        + f'<text x="{P}" y="{H-P+18}" font-size="11" fill="#666">iter {xmin}</text>'
        + f'<text x="{W-P}" y="{H-P+18}" font-size="11" fill="#666" text-anchor="end">iter {xmax}</text>'
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}">'
        f"{axis}"
        f'<polyline fill="none" stroke="#1f6feb" stroke-width="1.5" stroke-dasharray="3,3" points="{line_pts}"/>'
        f'<polyline fill="none" stroke="#1f8a3e" stroke-width="2.5" points="{cur_pts}"/>'
        f"{dots}"
        f"</svg>"
    )
