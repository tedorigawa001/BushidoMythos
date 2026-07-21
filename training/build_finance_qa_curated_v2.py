#!/usr/bin/env python3
"""Build the deterministic, family-disjoint Finance QA curated v2 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN = ROOT / "training/train_data/finance_qa_curated_v2_train.json"
DEFAULT_VALIDATION = ROOT / "training/eval_data/finance_qa_curated_v2_validation.json"
DEFAULT_HELD_OUT = ROOT / "training/eval_data/finance_qa_v2.json"
DEFAULT_MANIFEST = ROOT / "training/train_data/finance_qa_curated_v2_manifest.json"

INSTRUMENTS = (
    "equity index future", "government bond future", "gold future",
    "energy future", "major-currency pair", "investment-grade bond",
    "technology stock", "bank stock", "commodity ETF", "rate swap",
)
MARKETS = (
    "New York session", "London session", "Asian session", "month-end window",
    "quarter-end window", "thin holiday session", "high-volume morning",
)
LIMITS = ("0.25", "0.35", "0.50", "0.60", "0.75", "0.90", "1.00")
FACTORS = ("rates", "equities", "credit spreads", "the dollar", "oil", "volatility")
EVENTS = (
    "policy announcement", "inventory report", "debt auction", "index rebalance",
    "regulatory decision", "employment release", "inflation release",
)


PROSE_SPECS = {
    "train": (
        ("risk_management", "futures_risk_budget",
         "Before entering a {instrument} during the {market}, how should a trader enforce a {limit}% account-risk cap over {horizon}?",
         "For the {instrument}, first convert the {limit}% cap into a dollar loss budget. Set an invalidation level from the trade thesis, convert its distance into dollars per contract with the multiplier, include fees and stressed slippage, then divide and round down. Recheck margin and gap exposure for the {horizon} holding period, and skip the order if one contract exceeds the budget."),
        ("risk_management", "option_spread_limit",
         "A trader is considering an option spread on a {instrument} around a {event}. What controls keep the loss within {limit}% over {horizon}?",
         "Map the spread payoff and confirm the contractual maximum loss, assignment exposure, and any uncovered leg. Compare that stressed loss with the {limit}% account budget, include execution and early-assignment costs, and size from the worse amount. For the {event} over {horizon}, leave margin headroom and reject structures whose loss cannot be bounded reliably."),
        ("portfolio", "factor_concentration",
         "Several positions all respond to {factor} during the {market}. How should their combined exposure be controlled for {horizon}?",
         "Treat {factor} as a shared portfolio exposure rather than counting each ticket as independent. Aggregate directional and nonlinear sensitivities, estimate losses under normal and stressed correlations, and compare the total with portfolio limits. During the {market}, reduce overlapping positions or add a tested hedge, while allowing for basis risk over {horizon}."),
        ("portfolio", "portfolio_heat_stress",
         "A portfolio adds a {instrument} trade with a {limit}% loss budget while existing positions are vulnerable to {event}. What must be checked?",
         "Add the proposed {limit}% budget to current portfolio heat and group all positions exposed to the {event}. Stress simultaneous gaps, correlation increases, and weaker liquidity instead of summing only normal stop losses. Accept the {instrument} trade only if aggregate and factor-level limits remain intact with a cash and margin buffer."),
        ("decision_safety", "drawdown_governance",
         "A {instrument} strategy has exceeded its historical drawdown during the {market}. What is a disciplined response over {horizon}?",
         "Pause or reduce the {instrument} risk instead of increasing size to recover. Check data, implementation, costs, fills, and whether the {market} represents a new regime; then compare the loss with precommitted limits. Resume over {horizon} only after an independent review identifies a defensible cause and approves a smaller risk budget."),
        ("decision_safety", "forecast_uncertainty",
         "How should a forecast about {factor} after a {event} guide a {horizon} trade without claiming certainty?",
         "State a base case for {factor}, the evidence behind it, and alternative outcomes after the {event}. Define observations that would invalidate each scenario and update probabilities as evidence changes. Translate uncertainty into smaller size, monitoring, and exit rules for {horizon}; do not present the forecast as a guaranteed direction."),
        ("market_structure", "order_book_execution",
         "A large {instrument} order must be executed in the {market}. How should liquidity and price impact be managed?",
         "Estimate available depth, spread, participation rate, and how quickly the {instrument} book replenishes in the {market}. Split the order when urgency permits, set price and time limits, monitor realized impact, and stop if liquidity deteriorates. A displayed quote is not a guarantee for the full quantity, so size the original position for stressed exit cost."),
        ("market_structure", "partial_fill_management",
         "Only part of a {instrument} order fills before a {event}. What risks should be managed over {horizon}?",
         "Measure the exposure actually filled and cancel or revise stale orders before the {event}. Decide whether the residual position still fits the thesis, hedge only with an instrument whose basis risk is understood, and avoid chasing the remainder through a widening spread. Recalculate loss and margin limits for the partial position over {horizon}."),
        ("event_risk", "central_bank_gap_control",
         "A {instrument} position may be held through a central-bank {event}. Describe controls for a {limit}% risk cap.",
         "Model discontinuous price moves, spread widening, and stop slippage around the central-bank {event}; recent volatility alone is insufficient. Reduce the {instrument} quantity, use a bounded hedge only when its basis and liquidity are acceptable, and maintain excess margin. Avoid the event when a plausible gap would breach the {limit}% account-risk cap."),
        ("event_risk", "commodity_report_jump",
         "How should a {instrument} trader prepare for a scheduled {event} during the {market}?",
         "Identify the exact {event} time and compare consensus with alternative outcomes, but do not assume the surprise direction. Stress gaps, limit moves, spread changes, and correlated instruments for the {instrument}. Before the {market}, reduce quantity or flatten when the stressed loss exceeds limits, and verify that protective orders may fill away from their triggers."),
        ("macro", "yield_curve_repricing",
         "Explain how a {event} could reprice the yield curve and affect a {instrument} over {horizon}.",
         "The {event} can change expected policy rates, inflation compensation, term premium, or growth expectations, and different maturities need not move in parallel. Revalue the {instrument} under steepening, flattening, and level-shift scenarios over {horizon}. Frame the direction conditionally because positioning and prior expectations can dominate the initial reaction."),
        ("macro", "real_rate_currency_channel",
         "How can a change in real-rate expectations affect {factor} after a {event}, and why is the outcome conditional?",
         "Higher expected real returns can support assets exposed to {factor}, but only relative rates, inflation credibility, growth, positioning, and risk sentiment determine the net move. Compare the {event} with what was priced and with policy abroad. Use scenarios rather than a guaranteed direction because capital flows and safe-haven demand can offset the rate channel."),
        ("risk_metrics", "expected_shortfall_use",
         "How should expected shortfall for a {instrument} be interpreted at a {limit}% tail threshold over {horizon}?",
         "Expected shortfall estimates the average loss beyond the selected tail cutoff for the {instrument}; it describes severity after the threshold is crossed, not a maximum loss. Its reliability depends on sample, model, horizon, liquidity, and regime assumptions. Pair the {limit}% tail estimate with scenario tests and do not treat it as a guaranteed bound over {horizon}."),
        ("risk_metrics", "scenario_stress_design",
         "Design a stress test for positions exposed to {factor} and a {event} during the {market}.",
         "Choose severe but coherent shocks to {factor} and the {event}, let correlations and liquidity worsen together, and fully revalue nonlinear positions. Include funding, margin, spread, and liquidation effects during the {market}. Document assumptions and connect each loss level to actions such as reducing exposure, raising cash, or stopping new trades."),
    ),
    "validation": (
        ("risk_management", "short_sale_squeeze",
         "A short position in a {instrument} faces a possible {event}. How should its asymmetric risk be limited to {limit}%?",
         "A short {instrument} can lose more than its initial proceeds as price rises, while borrow recalls, fees, gaps, and the {event} can accelerate losses. Cap quantity from a stressed buy-in price rather than a nearby stop alone, monitor borrow and margin, and keep the modeled account loss below {limit}%. Exit when the thesis fails instead of averaging into a squeeze."),
        ("risk_management", "volatility_scaled_size",
         "Volatility in a {instrument} has risen sharply in the {market}. How should a {limit}% risk budget change position size?",
         "Keep the {limit}% dollar budget fixed, recalculate a structurally valid invalidation distance using the higher volatility, and divide by the larger per-unit risk. This normally produces a smaller {instrument} quantity. Add wider spread and slippage assumptions for the {market}, and skip the trade if the minimum lot would exceed the budget."),
        ("portfolio", "sector_overlap",
         "Two holdings in a {instrument} sector appear different but share exposure to {factor}. How should diversification be assessed?",
         "Look through the labels and estimate both holdings' sensitivity to {factor}, common funding, and crowded ownership. Calculate combined contribution to risk under normal and stressed correlations, not merely the number of securities. Reduce concentration or add genuinely different exposures when a single shock could impair both positions."),
        ("portfolio", "hedge_basis_risk",
         "A portfolio uses a {instrument} to hedge exposure to {factor} for {horizon}. What basis risks remain?",
         "The {instrument} and the exposure may respond differently as correlation, maturity, location, credit quality, or liquidity changes. Estimate hedge ratio uncertainty and stress divergence in {factor}, including roll and execution costs over {horizon}. Set limits on residual exposure and monitor the relationship instead of assuming the hedge is exact."),
        ("decision_safety", "model_drift_response",
         "A model trading a {instrument} has underperformed for {horizon} after a {event}. What review is appropriate?",
         "Reduce or pause the {instrument} allocation, verify inputs and implementation, and compare current feature and outcome distributions with training data after the {event}. Test plausible regime changes on untouched samples and include costs. Restore risk only through a documented approval process; recent losses alone do not justify increasing size."),
        ("decision_safety", "guaranteed_signal_claim",
         "A vendor says its {instrument} signal guarantees profit after every {event}. How should this claim be evaluated?",
         "Treat a guaranteed-profit claim as a red flag. Demand timestamped out-of-sample evidence, realistic costs, drawdowns, failure cases, and independent verification across multiple {event} regimes. Check selection and survivorship bias and never risk capital solely on the claim; all {instrument} strategies retain uncertainty and loss potential."),
        ("market_structure", "opening_auction",
         "A {instrument} order is planned for an opening auction after a {event}. What execution risks matter?",
         "Indicative auction prices and imbalance can change rapidly after the {event}; the final match may differ from continuous-market quotes. Set quantity and price constraints, monitor imbalance and cancellation rules, and plan for an unfilled residual. Include gap and post-open liquidity risk when sizing the {instrument} order."),
        ("market_structure", "stop_limit_tradeoff",
         "Compare stop-market and stop-limit protection for a {instrument} during a fast {market} move.",
         "After triggering, a stop-market order prioritizes execution but can fill far away in the fast {market}. A stop-limit caps price but may not execute, leaving the {instrument} exposure open. Neither guarantees the planned loss; choose from liquidity and urgency, size for gaps, and monitor any residual position."),
        ("event_risk", "election_weekend_gap",
         "A {instrument} position spans a weekend election. How should gap risk be controlled within {limit}%?",
         "Weekend results can move the next tradable {instrument} price beyond any protective trigger and can impair liquidity. Model multiple election outcomes and correlated market gaps, then reduce or close the position when stressed loss exceeds {limit}%. Options can bound some outcomes only if strike, expiry, and counterparty liquidity remain effective."),
        ("event_risk", "credit_rating_event",
         "A rating review may affect a {instrument} over {horizon}. What risks should be considered?",
         "A downgrade or outlook change can widen spreads, trigger mandate-driven selling, alter collateral terms, and reduce liquidity in the {instrument}. Compare possible decisions with market expectations, stress related issuers and hedges, and reduce exposure when the {horizon} loss or funding demand breaches limits. The announcement direction is not certain."),
        ("macro", "fiscal_supply_bonds",
         "How could heavier government issuance affect a {instrument} over {horizon}?",
         "More supply can raise the yield concession required by buyers and pressure the {instrument} price, but demand, central-bank policy, maturity mix, and prior expectations can offset it. Test parallel and curve-shape changes over {horizon}. Present the result as conditional scenarios, not a certain selloff."),
        ("macro", "terms_trade_currency",
         "How can a change in commodity export prices affect {factor} after a {event}?",
         "Improved export prices may support income, trade balances, and assets linked to {factor}, while weaker prices can reverse that channel. The response also depends on import costs, hedging, policy, global risk appetite, and what the {event} had already priced. Compare relative-country effects and avoid a deterministic currency forecast."),
        ("risk_metrics", "liquidity_adjusted_var",
         "Why might ordinary value at risk understate losses for a {instrument} during the {market}?",
         "Ordinary value at risk may use normal returns and a fixed holding period while ignoring widening spreads, market impact, and the time needed to liquidate the {instrument}. Adjust the horizon and costs for stressed depth in the {market}, then supplement the estimate with expected shortfall and explicit gap scenarios. It remains an estimate, not a loss ceiling."),
        ("risk_metrics", "concentration_limit_review",
         "How should a concentration limit for exposure to {factor} be tested around a {event}?",
         "Measure gross, net, and nonlinear sensitivity to {factor}, including indirect positions and contingent exposures. Stress the {event} with correlation, liquidity, and margin changes and compare the resulting loss with capital and cash buffers. A nominal position percentage alone is insufficient; set escalation and reduction actions before the limit is reached."),
    ),
}

HELD_OUT_FAMILIES = {
    "leverage_risk", "stop_loss_placement", "position_size_calculation",
    "overnight_position", "fed_inflation", "liquidity_risk", "losing_position",
    "earnings_event", "reward_risk_calculation", "diversification",
    "guaranteed_return", "var_limitations",
}


def _money(value: Decimal) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _params(index: int) -> dict:
    days = index + 1
    return {
        "instrument": INSTRUMENTS[index % len(INSTRUMENTS)],
        "market": MARKETS[(index * 3) % len(MARKETS)],
        "horizon": f"{days} trading day" if days == 1 else f"{days} trading days",
        "limit": LIMITS[(index * 5) % len(LIMITS)],
        "factor": FACTORS[(index * 5) % len(FACTORS)],
        "event": EVENTS[(index * 4) % len(EVENTS)],
    }


def _prose_example(split: str, spec: tuple, index: int) -> dict:
    category, family, question, answer = spec
    params = _params(index)
    prefix = (
        "Give a risk-controlled answer. " if index % 2 else
        "Explain the decision process. "
    )
    suffix = (
        " State the key limitation." if index % 3 == 0 else
        " Include the condition for rejecting the trade."
    )
    return {
        "id": f"{split}_{family}_{index:03d}",
        "split": split,
        "scenario_family": family,
        "template_id": f"{family}_v1",
        "category": category,
        "instruction": prefix + question.format(**params) + suffix,
        "response": (
            answer.format(**params)
            + f" Reassess these controls if conditions change within {params['horizon']}."
        ),
        "parameters": params,
    }


def _portfolio_heat_example(split: str, index: int) -> dict:
    account = Decimal(40000 + index * 1750)
    losses = [Decimal(90 + index * 3), Decimal(140 + index * 4), Decimal(70 + index * 2)]
    total = sum(losses)
    percent = (total / account * 100).quantize(Decimal("0.01"))
    family = "portfolio_heat_arithmetic"
    return {
        "id": f"{split}_{family}_{index:03d}", "split": split,
        "scenario_family": family, "template_id": f"{family}_v1",
        "category": "calculation",
        "instruction": (
            f"An account has ${_money(account)}. Three trades have planned losses of "
            f"${_money(losses[0])}, ${_money(losses[1])}, and ${_money(losses[2])}. "
            "Calculate total portfolio heat in dollars and as a percentage of equity."
        ),
        "response": (
            f"Total planned loss is ${_money(losses[0])} + ${_money(losses[1])} + "
            f"${_money(losses[2])} = ${_money(total)}. Dividing ${_money(total)} by "
            f"${_money(account)} gives {percent}% of equity. This assumes planned exits; "
            "correlated gaps, spread widening, and slippage can make realized heat larger."
        ),
        "calculation": {
            "formula": "portfolio_heat", "inputs": {
                "account": str(account), "losses": [str(value) for value in losses],
            }, "outputs": {"total_loss": str(total), "percent": str(percent)},
        },
    }


def _drawdown_recovery_example(split: str, index: int) -> dict:
    drawdown = Decimal("8") + Decimal(index) * Decimal("0.5")
    remaining = Decimal(100) - drawdown
    recovery = (drawdown / remaining * 100).quantize(Decimal("0.01"))
    family = "drawdown_recovery_arithmetic"
    return {
        "id": f"{split}_{family}_{index:03d}", "split": split,
        "scenario_family": family, "template_id": f"{family}_v1",
        "category": "calculation",
        "instruction": (
            f"A portfolio falls {drawdown}% from its starting value. What percentage gain "
            "on the remaining capital is required to recover to the starting value?"
        ),
        "response": (
            f"The portfolio retains {remaining}% of starting capital. The missing "
            f"{drawdown} points divided by {remaining} equals {recovery}%. Therefore a "
            f"{recovery}% gain on the reduced base is required. Recovery exceeds the "
            "drawdown percentage because the gain is earned on less capital."
        ),
        "calculation": {
            "formula": "drawdown_recovery", "inputs": {"drawdown_percent": str(drawdown)},
            "outputs": {"remaining_percent": str(remaining), "recovery_percent": str(recovery)},
        },
    }


def _break_even_example(split: str, index: int) -> dict:
    win = Decimal(80 + index * 7)
    loss = Decimal(45 + index * 3)
    rate = (loss / (win + loss) * 100).quantize(Decimal("0.01"))
    family = "break_even_payoff_arithmetic"
    return {
        "id": f"{split}_{family}_{index:03d}", "split": split,
        "scenario_family": family, "template_id": f"{family}_v1",
        "category": "calculation",
        "instruction": (
            f"Before costs, a strategy gains ${_money(win)} on a win and loses "
            f"${_money(loss)} on a loss. Calculate the break-even win rate."
        ),
        "response": (
            f"Let p be the win rate. Break-even requires ${_money(win)}p - "
            f"${_money(loss)}(1-p) = 0, so p = {_money(loss)} / "
            f"({_money(win)} + {_money(loss)}) = {rate}%. Costs and changing payoff "
            "distributions would raise or destabilize this threshold."
        ),
        "calculation": {
            "formula": "break_even_rate", "inputs": {"win": str(win), "loss": str(loss)},
            "outputs": {"win_rate_percent": str(rate)},
        },
    }


def _leverage_impact_example(split: str, index: int) -> dict:
    leverage = Decimal(2 + index % 7)
    move = Decimal("0.5") + Decimal(index % 9) * Decimal("0.25")
    impact = (leverage * move).quantize(Decimal("0.01"))
    family = "leverage_impact_arithmetic"
    return {
        "id": f"{split}_{family}_{index:03d}", "split": split,
        "scenario_family": family, "template_id": f"{family}_v1",
        "category": "calculation",
        "instruction": (
            f"Gross exposure is {leverage} times account equity. Ignoring financing and "
            f"liquidation, approximate the equity impact of a {move}% adverse asset move."
        ),
        "response": (
            f"The linear estimate is {leverage} times {move}%, which equals an "
            f"approximately {impact}% equity loss. It is not a worst-case bound because "
            "gaps, margin changes, slippage, and nonlinear instruments can increase loss."
        ),
        "calculation": {
            "formula": "leverage_impact", "inputs": {
                "leverage": str(leverage), "adverse_move_percent": str(move),
            }, "outputs": {"equity_impact_percent": str(impact)},
        },
    }


def _recompute(calculation: dict) -> dict:
    formula = calculation["formula"]
    inputs = calculation["inputs"]
    if formula == "portfolio_heat":
        account = Decimal(inputs["account"])
        total = sum(Decimal(value) for value in inputs["losses"])
        return {"total_loss": str(total), "percent": str((total / account * 100).quantize(Decimal("0.01")))}
    if formula == "drawdown_recovery":
        drawdown = Decimal(inputs["drawdown_percent"])
        remaining = Decimal(100) - drawdown
        return {"remaining_percent": str(remaining), "recovery_percent": str((drawdown / remaining * 100).quantize(Decimal("0.01")))}
    if formula == "break_even_rate":
        win, loss = Decimal(inputs["win"]), Decimal(inputs["loss"])
        return {"win_rate_percent": str((loss / (win + loss) * 100).quantize(Decimal("0.01")))}
    if formula == "leverage_impact":
        value = Decimal(inputs["leverage"]) * Decimal(inputs["adverse_move_percent"])
        return {"equity_impact_percent": str(value.quantize(Decimal("0.01")))}
    raise ValueError(f"unknown calculation formula: {formula}")


def build_examples(split: str, per_family: int) -> list[dict]:
    examples = []
    for spec in PROSE_SPECS[split]:
        examples.extend(_prose_example(split, spec, index) for index in range(per_family))
    calculation_builders = (
        (_portfolio_heat_example, _drawdown_recovery_example)
        if split == "train" else
        (_break_even_example, _leverage_impact_example)
    )
    for builder in calculation_builders:
        examples.extend(builder(split, index) for index in range(per_family))
    return examples


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def validate_corpus(train: list[dict], validation: list[dict], held_out_path: Path) -> dict:
    all_examples = train + validation
    ids = [example["id"] for example in all_examples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate curated example id")
    for field in ("instruction", "response"):
        values = [_normalize(example[field]) for example in all_examples]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate curated {field}")
    train_families = {example["scenario_family"] for example in train}
    validation_families = {example["scenario_family"] for example in validation}
    if train_families & validation_families:
        raise ValueError("train/validation scenario-family leakage")
    if (train_families | validation_families) & HELD_OUT_FAMILIES:
        raise ValueError("curated/final-held-out scenario-family leakage")
    calculations = [example for example in all_examples if "calculation" in example]
    for example in calculations:
        expected = example["calculation"]["outputs"]
        if _recompute(example["calculation"]) != expected:
            raise ValueError(f"calculation mismatch: {example['id']}")
        if any(value not in example["response"] for value in expected.values()):
            raise ValueError(f"calculation output absent from response: {example['id']}")
    held_out_raw = held_out_path.read_bytes()
    held_out = json.loads(held_out_raw.decode("utf-8"))
    held_out_ids = {case["id"] for case in held_out["cases"]}
    if held_out_ids != HELD_OUT_FAMILIES:
        raise ValueError("held-out family registry no longer matches evaluation case ids")
    return {
        "counts": {"train": len(train), "validation": len(validation), "held_out": len(held_out["cases"])},
        "categories": {
            "train": dict(sorted(Counter(item["category"] for item in train).items())),
            "validation": dict(sorted(Counter(item["category"] for item in validation).items())),
        },
        "scenario_families": {
            "train": sorted(train_families),
            "validation": sorted(validation_families),
            "held_out": sorted(HELD_OUT_FAMILIES),
        },
        "calculation_examples": len(calculations),
        "held_out": {
            "path": str(held_out_path.relative_to(ROOT)),
            "version": held_out.get("version"),
            "sha256": hashlib.sha256(held_out_raw).hexdigest(),
        },
        "checks": {
            "unique_ids": True, "unique_instructions": True, "unique_responses": True,
            "family_disjoint": True, "calculations_recomputed": True,
        },
    }


def _payload(split: str, examples: list[dict]) -> dict:
    return {
        "version": f"finance_qa_curated_v2_{split}",
        "split": split,
        "description": "Project-authored deterministic Finance QA SFT scenarios; family-disjoint from validation and final held-out evaluation.",
        "license": "Project-authored synthetic instructional data",
        "examples": examples,
    }


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_and_write(args: argparse.Namespace) -> dict:
    train = build_examples("train", args.train_per_family)
    validation = build_examples("validation", args.validation_per_family)
    manifest = validate_corpus(train, validation, args.held_out)
    train_sha = _write_json(args.train_out, _payload("train", train))
    validation_sha = _write_json(args.validation_out, _payload("validation", validation))
    manifest.update({
        "version": "finance_qa_curated_v2",
        "generator": str(Path(__file__).relative_to(ROOT)),
        "outputs": {
            "train": {"path": _display_path(args.train_out), "sha256": train_sha},
            "validation": {"path": _display_path(args.validation_out), "sha256": validation_sha},
        },
    })
    _write_json(args.manifest_out, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation_out", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--held_out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--manifest_out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train_per_family", type=int, default=40)
    parser.add_argument("--validation_per_family", type=int, default=10)
    args = parser.parse_args()
    if args.train_per_family <= 0 or args.validation_per_family <= 0:
        parser.error("per-family counts must be positive")
    if not args.held_out.is_file():
        parser.error(f"held-out suite not found: {args.held_out}")
    return args


def main() -> None:
    args = parse_args()
    manifest = build_and_write(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
