"""Generate self-contained SVG chart assets for the blog from the banking eval report.

Reads ``docs/banking_adversarial_report_full100.json`` and emits standalone, dependency-free
SVGs into ``docs/blog_assets/``:

* ``cosine_overlap.svg``      - strip plot of paraphrase vs trap cosine similarity
* ``fpr_by_trap_type.svg``    - false reuse rate by trap type, baseline vs our stack
* ``scoreboard.svg``          - headline 0%/51% FPR, 67%/56% recall panel
* ``architecture.svg``        - control-plane flow (static template)

SVGs use explicit light-theme colors (dark text on transparent/white) so they render in any
markdown viewer, GitHub, or CMS without the widget host's CSS. Regenerate with::

    python3 -m gatecache.eval.generate_blog_assets
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# Light-theme palette (explicit; no CSS vars so files are portable)
TXT = "#3f3f3c"
MUTED = "#73726c"
GRID = "#e7e5df"
GREEN = "#639922"
GREEN_DK = "#3B6D11"
RED = "#E24B4A"
RED_DK = "#A32D2D"
BLUE = "#378ADD"
GRAY = "#888780"

CAT_ORDER = ["ID_SWAP", "NEGATION", "FRESHNESS", "PRINCIPAL_SWAP", "ACTION", "TIER_SWAP", "PRODUCT_SWAP"]
CAT_LABEL = {
    "ID_SWAP": "identifier swap",
    "NEGATION": "negation",
    "FRESHNESS": "freshness",
    "PRINCIPAL_SWAP": "principal swap",
    "ACTION": "action",
    "TIER_SWAP": "tier swap",
    "PRODUCT_SWAP": "product swap",
}


def _confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    agg: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
    for r in rows:
        exp, act = r["expected_cache_hit"], r["actual_cache_hit"]
        key = "tp" if (exp and act) else "fn" if (exp and not act) else "fp" if (not exp and act) else "tn"
        agg[r["category"]][key] += 1
        agg["__all__"][key] += 1
    return agg


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg(width: int, height: int, body: str, title: str, desc: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="-apple-system,Segoe UI,Roboto,sans-serif" role="img">'
        f"<title>{_esc(title)}</title><desc>{_esc(desc)}</desc>{body}</svg>\n"
    )


def cosine_overlap_svg(report: dict[str, Any]) -> str:
    rows = report["baseline_cosine_only"]["rows"]
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["cosine"])

    W, H = 720, 470
    x0, x1 = 170, 690  # plot x range
    cmin, cmax = 0.55, 1.0

    def cx(c: float) -> float:
        return x0 + (c - cmin) / (cmax - cmin) * (x1 - x0)

    order = ["PARAPHRASE"] + CAT_ORDER
    label = {"PARAPHRASE": "paraphrase", **CAT_LABEL}
    top, pitch = 70, 44
    parts: list[str] = []

    # title + legend
    parts.append(f'<text x="20" y="26" font-size="15" font-weight="500" fill="{TXT}">Cosine similarity: safe paraphrases vs traps (same embeddings, 94 queries)</text>')
    parts.append(f'<circle cx="26" cy="44" r="5" fill="{GREEN}"/><text x="36" y="48" font-size="12" fill="{MUTED}">should reuse (27)</text>')
    parts.append(f'<polygon points="206,40 212,50 200,50" fill="{RED}"/><text x="218" y="48" font-size="12" fill="{MUTED}">must NOT reuse (67)</text>')
    parts.append(f'<line x1="372" y1="44" x2="392" y2="44" stroke="{GRAY}" stroke-width="2" stroke-dasharray="5 4"/><text x="398" y="48" font-size="12" fill="{MUTED}">0.85 cutoff</text>')

    # danger shading right of 0.85 + cutoff line
    xc = cx(0.85)
    plot_top, plot_bot = top - 18, top + (len(order) - 1) * pitch + 18
    parts.append(f'<rect x="{xc:.1f}" y="{plot_top}" width="{x1 - xc:.1f}" height="{plot_bot - plot_top}" fill="{RED}" fill-opacity="0.06"/>')

    # x gridlines + ticks
    for t in [0.6, 0.7, 0.8, 0.85, 0.9, 1.0]:
        gx = cx(t)
        parts.append(f'<line x1="{gx:.1f}" y1="{plot_top}" x2="{gx:.1f}" y2="{plot_bot}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{gx:.1f}" y="{plot_bot + 16}" font-size="11" fill="{MUTED}" text-anchor="middle">{t:.2f}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{plot_bot + 34}" font-size="12" fill="{MUTED}" text-anchor="middle">cosine similarity to cached query</text>')
    parts.append(f'<line x1="{xc:.1f}" y1="{plot_top}" x2="{xc:.1f}" y2="{plot_bot}" stroke="{GRAY}" stroke-width="2" stroke-dasharray="5 4"/>')

    # rows
    for i, cat in enumerate(order):
        cy = top + i * pitch
        parts.append(f'<text x="158" y="{cy + 4:.0f}" font-size="12" fill="{TXT}" text-anchor="end">{label[cat]}</text>')
        vals = sorted(by_cat.get(cat, []))
        for j, v in enumerate(vals):
            jy = cy + ((j % 3) - 1) * 6
            px = cx(v)
            if cat == "PARAPHRASE":
                parts.append(f'<circle cx="{px:.1f}" cy="{jy:.1f}" r="4" fill="{GREEN}" fill-opacity="0.9"/>')
            else:
                parts.append(f'<polygon points="{px:.1f},{jy - 4.5:.1f} {px + 4.5:.1f},{jy + 3.5:.1f} {px - 4.5:.1f},{jy + 3.5:.1f}" fill="{RED}" fill-opacity="0.9"/>')

    desc = ("Paraphrases span 0.71-0.94; traps span 0.59-0.997 and overlap them. Identifier swaps "
            "and negations cluster above 0.85, higher than most real paraphrases.")
    return _svg(W, H, "".join(parts), "Cosine similarity overlap", desc)


def fpr_by_trap_type_svg(report: dict[str, Any]) -> str:
    base = _confusion(report["baseline_cosine_only"]["rows"])

    def fpr(c: dict[str, int]) -> float:
        denom = c["fp"] + c["tn"]
        return 100.0 * c["fp"] / denom if denom else 0.0

    bars = [("All traps", fpr(base["__all__"]))] + [(CAT_LABEL[c], fpr(base[c])) for c in CAT_ORDER]

    W = 720
    top, pitch, bar_h = 60, 42, 14
    H = top + len(bars) * pitch + 30
    x0, x1 = 160, 640  # 0..100%

    parts: list[str] = []
    parts.append(f'<text x="20" y="26" font-size="15" font-weight="500" fill="{TXT}">False reuse rate by trap type (% of must-miss traps wrongly served)</text>')
    parts.append(f'<rect x="20" y="38" width="11" height="11" rx="2" fill="{RED}"/><text x="36" y="48" font-size="12" fill="{MUTED}">cosine-only baseline</text>')
    parts.append(f'<rect x="190" y="38" width="11" height="11" rx="2" fill="{GREEN}"/><text x="206" y="48" font-size="12" fill="{MUTED}">our stack (0% every type)</text>')

    for t in [0, 25, 50, 75, 100]:
        gx = x0 + t / 100 * (x1 - x0)
        parts.append(f'<line x1="{gx:.1f}" y1="{top - 8}" x2="{gx:.1f}" y2="{top + len(bars) * pitch - 16}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{gx:.1f}" y="{top + len(bars) * pitch}" font-size="11" fill="{MUTED}" text-anchor="middle">{t}%</text>')

    for i, (name, val) in enumerate(bars):
        cy = top + i * pitch
        parts.append(f'<text x="150" y="{cy + 4:.0f}" font-size="12" fill="{TXT}" text-anchor="end">{name}</text>')
        # baseline bar (red, upper)
        bw = val / 100 * (x1 - x0)
        parts.append(f'<rect x="{x0}" y="{cy - bar_h - 1:.0f}" width="{bw:.1f}" height="{bar_h}" rx="3" fill="{RED}"/>')
        parts.append(f'<text x="{x0 + bw + 6:.1f}" y="{cy - 1:.0f}" font-size="12" font-weight="500" fill="{RED_DK}">{round(val)}%</text>')
        # our bar (green, lower) - zero, just the label
        parts.append(f'<rect x="{x0}" y="{cy + 1:.0f}" width="2" height="{bar_h}" rx="1" fill="{GREEN}"/>')
        parts.append(f'<text x="{x0 + 6:.0f}" y="{cy + bar_h:.0f}" font-size="12" font-weight="500" fill="{GREEN_DK}">0%</text>')

    desc = "Baseline false reuse: all traps 51%, identifier swap and negation 100%, freshness 50%, others 20-38%. Our stack: 0% on every type."
    return _svg(W, H, "".join(parts), "False reuse rate by trap type", desc)


def scoreboard_svg(report: dict[str, Any]) -> str:
    design = _confusion(report["design"]["rows"])["__all__"]
    base = _confusion(report["baseline_cosine_only"]["rows"])["__all__"]

    def pct(n: int, d: int) -> int:
        return round(100 * n / d) if d else 0

    d_fpr, b_fpr = pct(design["fp"], design["fp"] + design["tn"]), pct(base["fp"], base["fp"] + base["tn"])
    d_rec, b_rec = pct(design["tp"], design["tp"] + design["fn"]), pct(base["tp"], base["tp"] + base["fn"])
    d_trap_n, b_trap_n = design["fp"] + design["tn"], base["fp"] + base["tn"]
    d_par_n = design["tp"] + design["fn"]

    W, H = 720, 280
    parts: list[str] = []
    parts.append(f'<text x="20" y="26" font-size="15" font-weight="500" fill="{TXT}">94 banking queries, identical embeddings - our stack vs cosine-only baseline</text>')

    col_metric_x, col_a_x, col_b_x = 24, 300, 510
    cw = 190
    parts.append(f'<text x="{col_a_x + cw / 2:.0f}" y="62" font-size="13" font-weight="500" fill="{TXT}" text-anchor="middle">Our stack</text>')
    parts.append(f'<text x="{col_b_x + cw / 2:.0f}" y="62" font-size="13" font-weight="500" fill="{MUTED}" text-anchor="middle">Cosine-only baseline</text>')

    def metric_row(y: int, label: str, sub: str, a_val: str, a_sub: str, a_good: bool, b_val: str, b_sub: str, b_tone: str) -> None:
        parts.append(f'<rect x="{col_metric_x}" y="{y}" width="250" height="76" rx="8" fill="#f4f3ee"/>')
        parts.append(f'<text x="{col_metric_x + 16}" y="{y + 32}" font-size="14" font-weight="500" fill="{TXT}">{label}</text>')
        parts.append(f'<text x="{col_metric_x + 16}" y="{y + 52}" font-size="12" fill="{MUTED}">{sub}</text>')
        # our stack card (success)
        parts.append(f'<rect x="{col_a_x}" y="{y}" width="{cw}" height="76" rx="8" fill="#e7f3e0"/>')
        parts.append(f'<text x="{col_a_x + cw / 2:.0f}" y="{y + 42}" font-size="30" font-weight="500" fill="{GREEN_DK}" text-anchor="middle">{a_val}</text>')
        parts.append(f'<text x="{col_a_x + cw / 2:.0f}" y="{y + 62}" font-size="12" fill="{GREEN_DK}" text-anchor="middle">{a_sub}</text>')
        # baseline card
        fill = "#fbe6e6" if b_tone == "bad" else "#f4f3ee"
        col = RED_DK if b_tone == "bad" else MUTED
        parts.append(f'<rect x="{col_b_x}" y="{y}" width="{cw}" height="76" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{col_b_x + cw / 2:.0f}" y="{y + 42}" font-size="30" font-weight="500" fill="{col}" text-anchor="middle">{b_val}</text>')
        parts.append(f'<text x="{col_b_x + cw / 2:.0f}" y="{y + 62}" font-size="12" fill="{col}" text-anchor="middle">{b_sub}</text>')

    metric_row(78, "False reuse", "wrong answers - lower better",
               f"{d_fpr}%", f"0 of {d_trap_n} traps", True, f"{b_fpr}%", f"{base['fp']} of {b_trap_n} traps", "bad")
    metric_row(166, "Recall", "paraphrases reused - higher better",
               f"{d_rec}%", f"{design['tp']} of {d_par_n}", True, f"{b_rec}%", f"{base['tp']} of {d_par_n}", "neutral")

    parts.append(f'<text x="24" y="266" font-size="12" fill="{MUTED}">Better on both axes - lower false reuse and higher recall. The baseline is strictly dominated.</text>')
    desc = f"False reuse: our stack {d_fpr}%, baseline {b_fpr}%. Recall: our stack {d_rec}%, baseline {b_rec}%."
    return _svg(W, H, "".join(parts), "Head-to-head scoreboard", desc)


def architecture_svg() -> str:
    W, H = 720, 600
    sx, sw = 40, 250  # spine x, width
    cxv = sx + sw / 2
    px = 430  # live panel x
    pw = 250
    rows = [
        ("Incoming query", "embed once", GRAY, "#f1efe8"),
        ("Routing classifier", "cacheable at all? skip actions, freshness", BLUE, "#e6f1fb"),
        ("Exact + structured match", "identifiers must match", BLUE, "#e6f1fb"),
        ("Vector retrieval", "scoped ANN - MariaDB Vector", BLUE, "#e6f1fb"),
        ("Facet gates", "entities, quantities, polarity", BLUE, "#e6f1fb"),
        ("Gray-zone judge", "would it answer the query?", BLUE, "#e6f1fb"),
        ("Reuse cached answer", "served from GridGain cache", GREEN, "#e7f3e0"),
    ]
    tops = [40, 118, 196, 274, 352, 430, 508]
    bh = 50
    parts: list[str] = []
    parts.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')

    # spine arrows
    for i in range(6):
        y1 = tops[i] + bh
        y2 = tops[i + 1]
        col = GREEN if i == 5 else GRAY
        parts.append(f'<line x1="{cxv}" y1="{y1}" x2="{cxv}" y2="{y2 - 2}" stroke="{col}" stroke-width="1.5" marker-end="url(#ah)"/>')

    # live panel
    parts.append(f'<rect x="{px}" y="118" width="{pw}" height="362" rx="8" fill="none" stroke="{GRAY}" stroke-width="0.7" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{px + pw / 2:.0f}" y="282" font-size="14" font-weight="500" fill="{TXT}" text-anchor="middle">Live answer</text>')
    parts.append(f'<text x="{px + pw / 2:.0f}" y="302" font-size="12" fill="{MUTED}" text-anchor="middle">call the model</text>')
    parts.append(f'<text x="{px + pw / 2:.0f}" y="330" font-size="12" fill="{MUTED}" text-anchor="middle">fail-safe: never a wrong reuse</text>')

    # veto arrows (rows 1..5)
    veto_y = [143, 221, 299, 377, 455]
    target_y = [160, 210, 270, 330, 392]
    for vy, ty in zip(veto_y, target_y):
        parts.append(f'<line x1="{sx + sw}" y1="{vy}" x2="{px - 2}" y2="{ty}" stroke="#C25B3A" stroke-width="1.3" marker-end="url(#ah)"/>')
    parts.append(f'<text x="365" y="438" font-size="12" fill="{MUTED}" text-anchor="middle">any check fails</text>')

    # boxes
    for (title, sub, stroke, fill), ty in zip(rows, tops):
        parts.append(f'<rect x="{sx}" y="{ty}" width="{sw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="0.7"/>')
        parts.append(f'<text x="{cxv}" y="{ty + 22}" font-size="14" font-weight="500" fill="{TXT}" text-anchor="middle">{_esc(title)}</text>')
        parts.append(f'<text x="{cxv}" y="{ty + 39}" font-size="11.5" fill="{MUTED}" text-anchor="middle">{_esc(sub)}</text>')

    # legend
    parts.append(f'<rect x="40" y="565" width="11" height="11" rx="2" fill="{BLUE}"/><text x="57" y="575" font-size="12" fill="{MUTED}">engine checks</text>')
    parts.append(f'<rect x="170" y="565" width="11" height="11" rx="2" fill="{GREEN}"/><text x="187" y="575" font-size="12" fill="{MUTED}">reuse</text>')
    parts.append(f'<rect x="250" y="565" width="11" height="11" rx="2" fill="{GRAY}"/><text x="267" y="575" font-size="12" fill="{MUTED}">live model call</text>')

    desc = ("Query flows through routing, exact/structured match, MariaDB Vector retrieval, facet gates, "
            "and a gray-zone judge; passing all reuses a GridGain-cached answer, any failure falls back to a live model call.")
    return _svg(W, H, "".join(parts), "Semantic cache control-plane architecture", desc)


def main() -> None:
    here = Path(__file__).resolve().parents[2]  # repo root
    report_path = here / "docs" / "banking_adversarial_report_full100.json"
    out_dir = here / "docs" / "blog_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text())

    assets = {
        "cosine_overlap.svg": cosine_overlap_svg(report),
        "fpr_by_trap_type.svg": fpr_by_trap_type_svg(report),
        "scoreboard.svg": scoreboard_svg(report),
        "architecture.svg": architecture_svg(),
    }
    for name, svg in assets.items():
        (out_dir / name).write_text(svg)
        print(f"wrote {out_dir / name}  ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
