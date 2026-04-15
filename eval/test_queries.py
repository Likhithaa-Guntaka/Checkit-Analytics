"""
eval/test_queries.py — Execution Plan Evaluation Framework for the Checkit RAG system.

Runs 20 real analyst queries and measures five core metrics drawn from the
execution plan quality targets:

  Metric               Formula                            Target
  ─────────────────────────────────────────────────────────────────
  Grounding Accuracy   evidence_count / key_points        >= 75%
  Hallucination Rate   unsupported / total (per-query)    <= 20%
  Reasoning Score      high=5 / medium=3 / low=1          >= 3.5
  Consistency          aligned=1.0 / mixed=0.5 / conflict  >= 65%
  Completeness         key_points / 3 (expect ≥ 3 pts)    >= 70%

Composite Score (0–100):
  0.30 × Grounding  + 0.20 × Reasoning/5  + 0.15 × Consistency
  + 0.15 × Completeness  + 0.10 × Citation  + 0.10 × (1 − Hallucination)

Overall PASS: 4 or more of the 5 metrics meet their target.

Run with:
    python eval/test_queries.py

Results are saved to eval/results.json.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reasoning.answerer import full_pipeline

# ── Query definitions ──────────────────────────────────────────────────────────

QUERIES: list[dict] = [
    # ── Single company (1–5) ───────────────────────────────────────────────────
    {
        "id":       1,
        "category": "Single Company",
        "query":    "What did NVIDIA say about data center demand in their latest earnings call?",
        "ticker":   "NVDA",
    },
    {
        "id":       2,
        "category": "Single Company",
        "query":    "What is Tesla's guidance on vehicle delivery growth?",
        "ticker":   "TSLA",
    },
    {
        "id":       3,
        "category": "Single Company",
        "query":    "How did Eli Lilly describe Mounjaro and Zepbound revenue performance?",
        "ticker":   None,   # LLY may not be in collection — no filter, broad search
    },
    {
        "id":       4,
        "category": "Single Company",
        "query":    "What did Airbnb say about international expansion?",
        "ticker":   "ABNB",
    },
    {
        "id":       5,
        "category": "Single Company",
        "query":    "How does Palantir describe their US commercial growth?",
        "ticker":   "PLTR",
    },
    # ── Cross-company comparison (6–10) ────────────────────────────────────────
    {
        "id":       6,
        "category": "Cross-Company Comparison",
        "query":    "Compare Datadog and CrowdStrike on revenue growth and guidance",
        "ticker":   None,
    },
    {
        "id":       7,
        "category": "Cross-Company Comparison",
        "query":    "How are Snowflake and MongoDB growing compared to each other?",
        "ticker":   None,
    },
    {
        "id":       8,
        "category": "Cross-Company Comparison",
        "query":    "What did Shopify and Booking Holdings say about consumer spending trends?",
        "ticker":   None,
    },
    {
        "id":       9,
        "category": "Cross-Company Comparison",
        "query":    "Compare ServiceNow and Workday on AI product adoption",
        "ticker":   None,
    },
    {
        "id":       10,
        "category": "Cross-Company Comparison",
        "query":    "How are Cloudflare and Veeva Systems describing their competitive moat?",
        "ticker":   None,
    },
    # ── Risk and challenges (11–15) ────────────────────────────────────────────
    {
        "id":       11,
        "category": "Risk & Challenges",
        "query":    "What macro risks did companies mention most frequently?",
        "ticker":   None,
    },
    {
        "id":       12,
        "category": "Risk & Challenges",
        "query":    "What did Microsoft say about Azure growth slowdown risks?",
        "ticker":   "MSFT",
    },
    {
        "id":       13,
        "category": "Risk & Challenges",
        "query":    "Which companies mentioned pricing pressure in their earnings calls?",
        "ticker":   None,
    },
    {
        "id":       14,
        "category": "Risk & Challenges",
        "query":    "What did Intuitive Surgical say about procedure volume risks?",
        "ticker":   None,   # ISRG may not be in collection
    },
    {
        "id":       15,
        "category": "Risk & Challenges",
        "query":    "What headwinds did Uber mention for their business?",
        "ticker":   "UBER",
    },
    # ── Sentiment and outlook (16–20) ──────────────────────────────────────────
    {
        "id":       16,
        "category": "Sentiment & Outlook",
        "query":    "Which companies gave the most positive guidance for next quarter?",
        "ticker":   None,
    },
    {
        "id":       17,
        "category": "Sentiment & Outlook",
        "query":    "What did management teams say about AI investment returns?",
        "ticker":   None,
    },
    {
        "id":       18,
        "category": "Sentiment & Outlook",
        "query":    "How did CEOs describe the demand environment overall?",
        "ticker":   None,
    },
    {
        "id":       19,
        "category": "Sentiment & Outlook",
        "query":    "What did MongoDB say about their outlook and customer expansion?",
        "ticker":   "MDB",
    },
    {
        "id":       20,
        "category": "Sentiment & Outlook",
        "query":    "Which companies mentioned hiring or headcount changes?",
        "ticker":   None,
    },
]

# ── Output helpers ─────────────────────────────────────────────────────────────

WIDTH = 70

_CONF_SYMBOL = {"high": "●", "medium": "◑", "low": "○"}
_SENT_SYMBOL = {"positive": "↑", "negative": "↓", "mixed": "~", "neutral": "–"}

_CAT_ABBREV = {
    "Single Company":           "Single",
    "Cross-Company Comparison": "Cross ",
    "Risk & Challenges":        "Risk  ",
    "Sentiment & Outlook":      "Outllk",
}


def _divider(char: str = "─") -> None:
    print(char * WIDTH)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(result: dict) -> dict:
    """
    Compute all execution-plan metrics for a single query result.

    Parameters
    ----------
    result : dict returned by full_pipeline() (or the error fallback).
             Must contain: evidence, key_points, confidence, consistency.

    Returns
    -------
    Dict with keys:
      grounding_score, grounded, hallucination_risk,
      reasoning_score, consistency_score, completeness_score,
      citation_score, composite_score (0–100).
    """
    evidence       = result.get("evidence", []) or []
    evidence_count = len(evidence)
    key_points     = result.get("key_points", []) or []
    confidence     = result.get("confidence", "low").lower()
    consistency    = result.get("consistency", "aligned").lower()

    # Grounding: evidence_count / max(key_points, 1), capped at 1.0
    grounding_score = min(evidence_count / max(len(key_points), 1), 1.0)

    # Hallucination risk: True when confidence is low AND no evidence
    hallucination_risk = confidence == "low" and evidence_count == 0

    # Reasoning: map confidence level to a 1–5 score
    reasoning_score = {"high": 5, "medium": 3, "low": 1}.get(confidence, 1)

    # Consistency: map the "consistency" field to a 0–1 score
    consistency_score = {"aligned": 1.0, "mixed": 0.5, "conflict": 0.0}.get(
        consistency, 1.0
    )

    # Completeness: expect at least 3 key points
    completeness_score = min(len(key_points) / 3, 1.0)

    # Citation: full credit when at least one evidence item is present
    citation_score = 1.0 if evidence_count > 0 else 0.0

    # Composite (weighted, 0–1 inputs; reasoning normalized to 0–1 via /5)
    composite = (
        0.30 * grounding_score
        + 0.20 * (reasoning_score / 5)
        + 0.15 * consistency_score
        + 0.15 * completeness_score
        + 0.10 * citation_score
        + 0.10 * (0.0 if hallucination_risk else 1.0)
    )

    return {
        "grounding_score":    round(grounding_score, 3),
        "grounded":           grounding_score >= 0.5,   # used for per-row display symbol
        "hallucination_risk": hallucination_risk,
        "reasoning_score":    reasoning_score,
        "consistency_score":  round(consistency_score, 3),
        "completeness_score": round(completeness_score, 3),
        "citation_score":     round(citation_score, 3),
        "composite_score":    round(composite * 100, 1),  # expressed as 0–100
    }


def compute_aggregate_metrics(results: list[dict]) -> dict:
    """
    Average all per-query metrics and check each against its execution-plan target.

    Parameters
    ----------
    results : list of row dicts produced by run_all() (each already has
              the fields from compute_metrics() merged in).

    Returns
    -------
    Dict with averages, pass/fail booleans, and overall_pass flag.
    """
    n = len(results)
    if n == 0:
        return {}

    avg_grounding    = sum(r["grounding_score"]    for r in results) / n
    halluc_rate      = sum(1 for r in results if r["hallucination_risk"]) / n
    avg_reasoning    = sum(r["reasoning_score"]    for r in results) / n
    avg_consistency  = sum(r["consistency_score"]  for r in results) / n
    avg_completeness = sum(r["completeness_score"] for r in results) / n
    avg_composite    = sum(r["composite_score"]    for r in results) / n

    pass_grounding    = avg_grounding    >= 0.75
    pass_hallucination = halluc_rate     <= 0.20
    pass_reasoning    = avg_reasoning    >= 3.5
    pass_consistency  = avg_consistency  >= 0.65
    pass_completeness = avg_completeness >= 0.70

    metrics_passed = sum([
        pass_grounding,
        pass_hallucination,
        pass_reasoning,
        pass_consistency,
        pass_completeness,
    ])

    return {
        "avg_grounding":       round(avg_grounding, 3),
        "hallucination_rate":  round(halluc_rate, 3),
        "avg_reasoning":       round(avg_reasoning, 2),
        "avg_consistency":     round(avg_consistency, 3),
        "avg_completeness":    round(avg_completeness, 3),
        "avg_composite":       round(avg_composite, 1),
        "pass_grounding":      pass_grounding,
        "pass_hallucination":  pass_hallucination,
        "pass_reasoning":      pass_reasoning,
        "pass_consistency":    pass_consistency,
        "pass_completeness":   pass_completeness,
        "metrics_passed":      metrics_passed,
        "overall_pass":        metrics_passed >= 4,
    }


# ── Row printer ────────────────────────────────────────────────────────────────

def _print_row(result: dict) -> None:
    """Print one compact result row."""
    qid       = result["id"]
    cat       = _CAT_ABBREV.get(result["category"], result["category"][:6])
    conf      = result["confidence"].lower()
    sent      = result["sentiment"].lower()
    n_ev      = result["evidence_count"]
    ans_len   = result["answer_length"]
    latency   = result["latency_s"]
    conf_sym  = _CONF_SYMBOL.get(conf, "?")
    sent_sym  = _SENT_SYMBOL.get(sent, "?")

    # Grounding indicator
    if result.get("hallucination_risk"):
        ground_sym = "⚠"
    elif result.get("grounded"):
        ground_sym = "✓"
    else:
        ground_sym = "–"

    query_short = result["query"][:42] + "…" if len(result["query"]) > 42 else result["query"]

    print(
        f"  {qid:>2}.  [{cat}]  {query_short:<43}  "
        f"{conf_sym} {conf:<6}  {sent_sym} {sent:<8}  "
        f"ev:{n_ev}  {ans_len:>3}w  {latency:.1f}s  {ground_sym}"
    )


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all() -> list[dict]:
    """Run all 20 queries and return a list of result dicts."""
    results: list[dict] = []

    print()
    _divider("═")
    print("  Checkit Analytics — Evaluation Suite (20 queries)")
    _divider("═")
    print(f"  {'#':>2}   {'Category':<8}  {'Query':<43}  "
          f"{'Conf':<8}  {'Sent':<9}  Ev  Len   Lat  Gnd")
    _divider()

    for spec in QUERIES:
        qid    = spec["id"]
        query  = spec["query"]
        ticker = spec.get("ticker")

        t_start = time.perf_counter()
        try:
            result_raw = full_pipeline(query, ticker=ticker)
        except Exception as exc:
            result_raw = {
                "answer":      f"ERROR: {exc}",
                "key_points":  [],
                "sentiment":   "neutral",
                "confidence":  "low",
                "evidence":    [],
                "limitations": str(exc),
                "risk_flags":  [],
                "consistency": "aligned",
            }
        latency = time.perf_counter() - t_start

        answer_text = result_raw.get("answer", "")
        evidence    = result_raw.get("evidence", []) or []
        key_points  = result_raw.get("key_points", []) or []
        confidence  = result_raw.get("confidence", "low")
        sentiment   = result_raw.get("sentiment", "neutral")

        metrics = compute_metrics(result_raw)

        row = {
            "id":               qid,
            "category":         spec["category"],
            "query":            query,
            "ticker_filter":    ticker,
            "confidence":       confidence,
            "sentiment":        sentiment,
            "evidence_count":   len(evidence),
            "answer_length":    len(answer_text.split()),
            "latency_s":        round(latency, 3),
            # ── execution-plan metrics ─────────────────────────────────────────
            "grounding_score":    metrics["grounding_score"],
            "grounded":           metrics["grounded"],
            "hallucination_risk": metrics["hallucination_risk"],
            "reasoning_score":    metrics["reasoning_score"],
            "consistency_score":  metrics["consistency_score"],
            "completeness_score": metrics["completeness_score"],
            "citation_score":     metrics["citation_score"],
            "composite_score":    metrics["composite_score"],
            # ── full answer payload ────────────────────────────────────────────
            "answer":       answer_text,
            "key_points":   key_points,
            "evidence":     evidence,
            "limitations":  result_raw.get("limitations", ""),
            "risk_flags":   result_raw.get("risk_flags", []),
            "consistency":  result_raw.get("consistency", "aligned"),
        }

        _print_row(row)
        results.append(row)

    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    agg = compute_aggregate_metrics(results)
    total = len(results)

    # Confidence breakdown
    high   = sum(1 for r in results if r["confidence"].lower() == "high")
    medium = sum(1 for r in results if r["confidence"].lower() == "medium")
    low    = sum(1 for r in results if r["confidence"].lower() == "low")
    avg_lat = sum(r["latency_s"] for r in results) / total if total else 0.0
    pass_rt = (high + medium) / total * 100 if total else 0.0

    # Category breakdown
    cat_counts: dict[str, dict] = {}
    for r in results:
        c = r["category"]
        if c not in cat_counts:
            cat_counts[c] = {"high": 0, "medium": 0, "low": 0}
        cat_counts[c][r["confidence"].lower()] += 1

    grounded_n = sum(1 for r in results if r.get("grounded"))
    halluc_n   = sum(1 for r in results if r.get("hallucination_risk"))

    print()
    _divider("═")
    print("  EVALUATION SUMMARY")
    _divider("═")
    print(f"  Total queries run        {total:>4}")
    print(f"  High confidence          {high:>4}  {'●' * high}")
    print(f"  Medium confidence        {medium:>4}  {'◑' * medium}")
    print(f"  Low confidence           {low:>4}  {'○' * low}")
    _divider()
    print(f"  Grounded answers         {grounded_n:>2}/{total}  {'✓' * grounded_n}")
    print(f"  Hallucination risk       {halluc_n:>2}/{total}  {'⚠' * halluc_n}")
    _divider()
    print(f"  Average latency          {avg_lat:>5.1f}s")
    print(f"  Pass rate (high+med)     {pass_rt:>5.1f}%")
    _divider()
    print("  Category breakdown:")
    for cat, counts in cat_counts.items():
        abbrev = _CAT_ABBREV.get(cat, cat[:16])
        h = counts["high"]
        m = counts["medium"]
        l = counts["low"]
        total_cat = h + m + l
        bar = ("●" * h) + ("◑" * m) + ("○" * l)
        print(f"    {abbrev}  pass {h + m}/{total_cat}  {bar}")

    _divider("═")
    print("  EXECUTION PLAN METRICS")
    _divider("═")

    def _metric_row(label: str, value: str, target: str, passed: bool) -> None:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {label:<28}  {value:>7}  (target {target})  {status}")

    _metric_row(
        "Grounding Accuracy",
        f"{agg['avg_grounding']*100:.1f}%",
        ">= 75%",
        agg["pass_grounding"],
    )
    _metric_row(
        "Hallucination Rate",
        f"{agg['hallucination_rate']*100:.1f}%",
        "<= 20%",
        agg["pass_hallucination"],
    )
    _metric_row(
        "Reasoning Score (1–5)",
        f"{agg['avg_reasoning']:.2f}",
        ">= 3.5",
        agg["pass_reasoning"],
    )
    _metric_row(
        "Cross-Source Consistency",
        f"{agg['avg_consistency']*100:.1f}%",
        ">= 65%",
        agg["pass_consistency"],
    )
    _metric_row(
        "Completeness",
        f"{agg['avg_completeness']*100:.1f}%",
        ">= 70%",
        agg["pass_completeness"],
    )
    _divider()
    print(f"  Composite Score          {agg['avg_composite']:>6.1f}/100")
    print(f"  Metrics passed           {agg['metrics_passed']}/5")
    _divider("═")
    overall_label = "OVERALL: PASS ✓" if agg["overall_pass"] else "OVERALL: FAIL ✗"
    print(f"  {overall_label}  ({agg['metrics_passed']}/5 metrics met target)")
    _divider("═")
    print()


# ── Save results ───────────────────────────────────────────────────────────────

def _save_results(results: list[dict]) -> Path:
    agg      = compute_aggregate_metrics(results)
    out_dir  = Path(__file__).resolve().parent
    out_file = out_dir / "results.json"

    payload = {
        "run_timestamp":        datetime.now(timezone.utc).isoformat(),
        "total_queries":        len(results),
        # ── confidence breakdown ───────────────────────────────────────────────
        "high_confidence":      sum(1 for r in results if r["confidence"].lower() == "high"),
        "medium_confidence":    sum(1 for r in results if r["confidence"].lower() == "medium"),
        "low_confidence":       sum(1 for r in results if r["confidence"].lower() == "low"),
        "grounded_count":       sum(1 for r in results if r.get("grounded")),
        "hallucination_risk_count": sum(1 for r in results if r.get("hallucination_risk")),
        "average_latency_s":    round(
            sum(r["latency_s"] for r in results) / len(results), 3
        ) if results else 0,
        "pass_rate_pct": round(
            sum(1 for r in results if r["confidence"].lower() in ("high", "medium"))
            / len(results) * 100,
            1,
        ) if results else 0,
        # ── execution plan aggregate metrics ──────────────────────────────────
        "execution_plan_metrics": {
            "avg_grounding_pct":       round(agg.get("avg_grounding", 0) * 100, 1),
            "hallucination_rate_pct":  round(agg.get("hallucination_rate", 0) * 100, 1),
            "avg_reasoning_score":     agg.get("avg_reasoning", 0),
            "avg_consistency_pct":     round(agg.get("avg_consistency", 0) * 100, 1),
            "avg_completeness_pct":    round(agg.get("avg_completeness", 0) * 100, 1),
            "avg_composite_score":     agg.get("avg_composite", 0),
            "pass_grounding":          agg.get("pass_grounding", False),
            "pass_hallucination":      agg.get("pass_hallucination", False),
            "pass_reasoning":          agg.get("pass_reasoning", False),
            "pass_consistency":        agg.get("pass_consistency", False),
            "pass_completeness":       agg.get("pass_completeness", False),
            "metrics_passed":          agg.get("metrics_passed", 0),
            "overall_pass":            agg.get("overall_pass", False),
        },
        "results": results,
    }

    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    results   = run_all()
    _print_summary(results)
    out_file  = _save_results(results)
    print(f"  Results saved → {out_file.relative_to(_ROOT)}")
    print()


if __name__ == "__main__":
    main()
