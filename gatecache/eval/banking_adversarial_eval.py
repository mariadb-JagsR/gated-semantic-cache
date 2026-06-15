"""Banking-domain adversarial cache eval: recall + false-positive-rate (FPR).

This is a holdout set, intentionally outside ``build_routing_dataset()``. It stresses the
distinction between *should-reuse* paraphrases (recall / TPR) and *must-not-reuse* traps
(false reuse / FPR): entity/id swaps, negation, destructive actions, freshness, tier swaps,
and principal swaps.

Two systems are scored on the SAME embeddings:

* ``design``   - the full classifier-first pipeline (routing + structured-exact + gates + judge)
* ``baseline`` - a LiteLLM/Redis-style cosine-only cache: embed seed + candidate, reuse iff
                 cosine >= ``baseline_threshold``. No routing, no judge, no structured match.

The headline number is FPR: of the must-not-reuse traps, what fraction did each system wrongly
serve from cache? A cosine-only cache cannot tell "order #A123" from "order #A456".

Each candidate's ``note`` is formatted ``"CATEGORY :: detail"`` so metrics can be broken out
per trap type.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from gatecache.eval.adversarial_cache_eval import (
    AdversarialCacheScenario,
    AdversarialCacheTest,
    run_adversarial_cache_eval,
)

# ---- category tags (parsed from the leading token of each candidate note) ----
PARAPHRASE = "PARAPHRASE"        # should reuse (recall positive)
PRODUCT_SWAP = "PRODUCT_SWAP"    # must not reuse
ID_SWAP = "ID_SWAP"             # must not reuse
NEGATION = "NEGATION"           # must not reuse
ACTION = "ACTION"               # must not reuse (destructive / state-mutating)
FRESHNESS = "FRESHNESS"         # must not reuse (value changes over time)
TIER_SWAP = "TIER_SWAP"         # must not reuse
PRINCIPAL_SWAP = "PRINCIPAL_SWAP"  # must not reuse (different account holder)


def _t(query: str, cache_hit: bool, category: str, detail: str) -> AdversarialCacheTest:
    return AdversarialCacheTest(query=query, cache_hit=cache_hit, note=f"{category} :: {detail}")


def _category_of(note: str) -> str:
    return note.split("::", 1)[0].strip()


def build_banking_adversarial_scenarios() -> list[AdversarialCacheScenario]:
    return [
        AdversarialCacheScenario(
            scenario="Minimum balance fee (policy paraphrase / product swap)",
            cached_query="What's the minimum balance to avoid the monthly fee on a basic checking account?",
            tests=(
                _t("How low can my checking balance go before I'm charged the monthly fee?", True, PARAPHRASE, "rephrase"),
                _t("What balance keeps my basic checking account fee-free?", True, PARAPHRASE, "rephrase"),
                _t("minimum balance to waive checking monthly fee", True, PARAPHRASE, "terse keyword form"),
                _t("What's the minimum balance to avoid the monthly fee on a basic savings account?", False, PRODUCT_SWAP, "checking->savings"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="ACH timing (timing paraphrase / product swap)",
            cached_query="How long does an ACH transfer take to clear?",
            tests=(
                _t("When will an ACH transfer post to my account?", True, PARAPHRASE, "rephrase"),
                _t("How many business days for an ACH transfer to settle?", True, PARAPHRASE, "rephrase"),
                _t("How long does a domestic wire transfer take to clear?", False, PRODUCT_SWAP, "ACH->wire"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="FDIC coverage (paraphrase / negation / product swap)",
            cached_query="Are my deposits automatically covered by FDIC insurance?",
            tests=(
                _t("Is FDIC coverage automatic on my deposit accounts?", True, PARAPHRASE, "rephrase"),
                _t("Do I need to enroll for FDIC insurance, or is it automatic?", True, PARAPHRASE, "rephrase"),
                _t("Are my deposits NOT covered by FDIC insurance?", False, NEGATION, "explicit negation"),
                _t("Are my brokerage investments automatically covered by FDIC insurance?", False, PRODUCT_SWAP, "deposits->investments (FDIC excludes)"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Wire transfer status (identifier swap)",
            cached_query="What's the status of wire transfer #WT-44219?",
            tests=(
                _t("Can you check the status of wire transfer WT-44219?", True, PARAPHRASE, "same id, rephrase"),
                _t("What's the status of wire transfer #WT-88301?", False, ID_SWAP, "different id"),
                _t("What's the status of wire transfer #WT-44291?", False, ID_SWAP, "digit transposition"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Dispute case lookup (single-digit identifier swap)",
            cached_query="Show the dispute details for case #D-7781.",
            tests=(
                _t("Pull up the details for dispute case D-7781.", True, PARAPHRASE, "same id, rephrase"),
                _t("Show the dispute details for case #D-7782.", False, ID_SWAP, "single-digit change"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Account holder lookup (principal swap)",
            cached_query="What's the account number on file for customer Jack Miller?",
            tests=(
                _t("Which account number do we have on file for Jack Miller?", True, PARAPHRASE, "same principal, rephrase"),
                _t("What's the account number on file for customer Jill Carter?", False, PRINCIPAL_SWAP, "different account holder"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Cancel Zelle payment (destructive action)",
            cached_query="Cancel my pending Zelle payment.",
            tests=(
                _t("Stop the Zelle transfer I just sent.", False, ACTION, "paraphrased action - both unsafe"),
                _t("Please cancel my pending wire transfer.", False, ACTION, "different action target"),
                _t("How do I cancel a Zelle payment?", False, ACTION, "how-to vs action - intent mismatch"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Lock stolen debit card (security action)",
            cached_query="Lock my debit card immediately, it was stolen.",
            tests=(
                _t("Freeze my debit card - I think it's compromised.", False, ACTION, "paraphrased action"),
                _t("Block my debit card right now.", False, ACTION, "paraphrased action"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Live balance (freshness-sensitive)",
            cached_query="What is my checking account balance right now?",
            tests=(
                _t("What's my current checking balance?", False, FRESHNESS, "value changes over time"),
                _t("What was my checking balance at close yesterday?", False, FRESHNESS, "temporal shift"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Mortgage rate (freshness-sensitive live rate)",
            cached_query="What's today's 30-year fixed mortgage rate?",
            tests=(
                _t("What is the current 30-year fixed mortgage rate?", False, FRESHNESS, "live rate, changes daily"),
                _t("What's today's 15-year fixed mortgage rate?", False, FRESHNESS, "term swap + live rate"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="International wire fee waiver (tier swap / negation)",
            cached_query="Are international wire transfer fees waived for premium accounts?",
            tests=(
                _t("Do premium accounts get free international wire transfers?", True, PARAPHRASE, "rephrase"),
                _t("Are international wire transfer fees waived for basic accounts?", False, TIER_SWAP, "premium->basic"),
                _t("Are international wire transfer fees NOT waived for premium accounts?", False, NEGATION, "explicit negation"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Overdraft eligibility (negation antonym)",
            cached_query="Which account types are eligible for overdraft protection?",
            tests=(
                _t("What accounts can enroll in overdraft protection?", True, PARAPHRASE, "rephrase"),
                _t("Which account types are ineligible for overdraft protection?", False, NEGATION, "eligible->ineligible"),
            ),
        ),
    ]


def build_banking_adversarial_scenarios_100() -> list[AdversarialCacheScenario]:
    """Rebalanced ~100-candidate suite: every trap type carried at n>=8 for meaningful
    per-type FPR. Should-reuse paraphrases (cache_hit=True) measure recall; all trap
    categories are cache_hit=False (must-miss)."""
    return [
        # ---------------- Fees, minimums, tiers ----------------
        AdversarialCacheScenario(
            scenario="Basic checking minimum balance (paraphrase / product / tier)",
            cached_query="What's the minimum balance to avoid the monthly fee on a basic checking account?",
            tests=(
                _t("How low can my checking balance go before the monthly fee kicks in?", True, PARAPHRASE, "rephrase"),
                _t("What balance keeps my basic checking account fee-free?", True, PARAPHRASE, "rephrase"),
                _t("minimum balance to waive the checking monthly maintenance fee", True, PARAPHRASE, "terse"),
                _t("What's the minimum balance to avoid the monthly fee on a basic savings account?", False, PRODUCT_SWAP, "checking->savings"),
                _t("What's the minimum balance to avoid the monthly fee on a premium checking account?", False, TIER_SWAP, "basic->premium"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Student checking maintenance fee (paraphrase / product / tier)",
            cached_query="What's the monthly maintenance fee on a student checking account?",
            tests=(
                _t("How much is the monthly fee for a student checking account?", True, PARAPHRASE, "rephrase"),
                _t("What's the monthly maintenance fee on a business checking account?", False, PRODUCT_SWAP, "student->business"),
                _t("What's the monthly maintenance fee on a premium checking account?", False, TIER_SWAP, "student->premium"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Overdraft fee waiver by tier (paraphrase / tier / negation)",
            cached_query="Are overdraft fees waived for premium checking customers?",
            tests=(
                _t("Do premium checking customers get overdraft fees waived?", True, PARAPHRASE, "rephrase"),
                _t("Are overdraft fees waived for basic checking customers?", False, TIER_SWAP, "premium->basic"),
                _t("Are overdraft fees NOT waived for premium checking customers?", False, NEGATION, "explicit negation"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Premium ATM withdrawal limit (paraphrase / tier)",
            cached_query="What's the ATM withdrawal limit on a premium checking account?",
            tests=(
                _t("How much can I withdraw at an ATM with premium checking?", True, PARAPHRASE, "rephrase"),
                _t("What's the ATM withdrawal limit on a basic checking account?", False, TIER_SWAP, "premium->basic"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Platinum lounge access (paraphrase / tier)",
            cached_query="Do platinum cardholders get free airport lounge access?",
            tests=(
                _t("Is airport lounge access free for platinum cardholders?", True, PARAPHRASE, "rephrase"),
                _t("Do gold cardholders get free airport lounge access?", False, TIER_SWAP, "platinum->gold"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Rewards card cashback rate (tier)",
            cached_query="What's the cashback rate on the premium rewards card?",
            tests=(
                _t("What's the cashback rate on the basic rewards card?", False, TIER_SWAP, "premium->basic"),
            ),
        ),
        # ---------------- Transfers & timing ----------------
        AdversarialCacheScenario(
            scenario="ACH clearing time (paraphrase / product)",
            cached_query="How long does an ACH transfer take to clear?",
            tests=(
                _t("When will an ACH transfer settle?", True, PARAPHRASE, "rephrase"),
                _t("How many business days for an ACH transfer to clear?", True, PARAPHRASE, "rephrase"),
                _t("How long does a domestic wire transfer take to clear?", False, PRODUCT_SWAP, "ACH->domestic wire"),
                _t("How long does an international wire take to clear?", False, PRODUCT_SWAP, "ACH->international wire"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Same-day ACH cutoff (paraphrase / product)",
            cached_query="What's the cutoff time for same-day ACH transfers?",
            tests=(
                _t("By what time must I submit an ACH transfer for same-day processing?", True, PARAPHRASE, "rephrase"),
                _t("What's the cutoff time for same-day wire transfers?", False, PRODUCT_SWAP, "ACH->wire"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Zelle daily limit (paraphrase / product / tier)",
            cached_query="What's the daily limit for Zelle transfers?",
            tests=(
                _t("How much can I send per day through Zelle?", True, PARAPHRASE, "rephrase"),
                _t("What's the daily limit for wire transfers?", False, PRODUCT_SWAP, "Zelle->wire"),
                _t("What's the daily Zelle limit for business accounts?", False, TIER_SWAP, "personal->business"),
            ),
        ),
        # ---------------- FDIC / insurance ----------------
        AdversarialCacheScenario(
            scenario="FDIC automatic coverage (paraphrase / negation / product)",
            cached_query="Are my deposits automatically covered by FDIC insurance?",
            tests=(
                _t("Is FDIC coverage automatic on my deposit accounts?", True, PARAPHRASE, "rephrase"),
                _t("Do I need to enroll for FDIC insurance, or is it automatic?", True, PARAPHRASE, "rephrase"),
                _t("Are my deposits NOT covered by FDIC insurance?", False, NEGATION, "explicit negation"),
                _t("Are my brokerage investments covered by FDIC insurance?", False, PRODUCT_SWAP, "deposits->investments"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="FDIC coverage limit (paraphrase / product)",
            cached_query="What's the FDIC coverage limit per depositor per bank?",
            tests=(
                _t("How much does FDIC insurance cover per depositor at one bank?", True, PARAPHRASE, "rephrase"),
                _t("What's the SIPC coverage limit per customer?", False, PRODUCT_SWAP, "FDIC->SIPC"),
            ),
        ),
        # ---------------- Identifier swaps ----------------
        AdversarialCacheScenario(
            scenario="Wire transfer status (identifier swap)",
            cached_query="What's the status of wire transfer #WT-44219?",
            tests=(
                _t("Can you check the status of wire transfer WT-44219?", True, PARAPHRASE, "same id, rephrase"),
                _t("What's the status of wire transfer #WT-88301?", False, ID_SWAP, "different id"),
                _t("What's the status of wire transfer #WT-44291?", False, ID_SWAP, "digit transposition"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Dispute case lookup (identifier swap)",
            cached_query="Show the dispute details for case #D-7781.",
            tests=(
                _t("Pull up the details for dispute case D-7781.", True, PARAPHRASE, "same id, rephrase"),
                _t("Show the dispute details for case #D-7782.", False, ID_SWAP, "single-digit change"),
                _t("Show the dispute details for case #D-7718.", False, ID_SWAP, "digit transposition"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Loan account rate (identifier swap)",
            cached_query="What's the interest rate on loan account LN-5567?",
            tests=(
                _t("What rate applies to loan account LN-5567?", True, PARAPHRASE, "same id, rephrase"),
                _t("What's the interest rate on loan account LN-5567A?", False, ID_SWAP, "suffix change"),
                _t("What's the interest rate on loan account LN-5576?", False, ID_SWAP, "digit transposition"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="CD maturity (identifier swap)",
            cached_query="When does CD account CD-3320 mature?",
            tests=(
                _t("What's the maturity date for CD account CD-3320?", True, PARAPHRASE, "same id, rephrase"),
                _t("When does CD account CD-3302 mature?", False, ID_SWAP, "digit transposition"),
                _t("When does CD account CD-3321 mature?", False, ID_SWAP, "single-digit change"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Branch routing number (identifier swap)",
            cached_query="What's the routing number for branch #0471?",
            tests=(
                _t("Which routing number belongs to branch 0471?", True, PARAPHRASE, "same id, rephrase"),
                _t("What's the routing number for branch #0472?", False, ID_SWAP, "single-digit change"),
            ),
        ),
        # ---------------- Negation / polarity ----------------
        AdversarialCacheScenario(
            scenario="Overdraft eligibility (paraphrase / negation)",
            cached_query="Which account types are eligible for overdraft protection?",
            tests=(
                _t("What accounts can enroll in overdraft protection?", True, PARAPHRASE, "rephrase"),
                _t("Which account types are ineligible for overdraft protection?", False, NEGATION, "eligible->ineligible"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="International wire fee waiver (paraphrase / negation / tier)",
            cached_query="Are international wire transfer fees waived for premium accounts?",
            tests=(
                _t("Do premium accounts get free international wire transfers?", True, PARAPHRASE, "rephrase"),
                _t("Are international wire transfer fees NOT waived for premium accounts?", False, NEGATION, "explicit negation"),
                _t("Are international wire transfer fees waived for basic accounts?", False, TIER_SWAP, "premium->basic"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Savings interest taxability (paraphrase / negation)",
            cached_query="Is the interest on my savings account taxable?",
            tests=(
                _t("Do I owe taxes on the interest from my savings account?", True, PARAPHRASE, "rephrase"),
                _t("Is the interest on my savings account tax-free?", False, NEGATION, "taxable->tax-free"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Foreign transaction fee (paraphrase / negation polarity)",
            cached_query="Does my credit card waive foreign transaction fees?",
            tests=(
                _t("Are foreign transaction fees waived on my credit card?", True, PARAPHRASE, "rephrase"),
                _t("Does my credit card charge foreign transaction fees?", False, NEGATION, "polarity flip"),
                _t("Does my credit card NOT waive foreign transaction fees?", False, NEGATION, "explicit negation"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Wire fraud protection coverage (paraphrase / negation)",
            cached_query="Are wire transfers covered by the bank's fraud protection?",
            tests=(
                _t("Does the bank's fraud protection cover wire transfers?", True, PARAPHRASE, "rephrase"),
                _t("Are wire transfers excluded from the bank's fraud protection?", False, NEGATION, "covered->excluded"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Mobile check deposit availability (negation)",
            cached_query="Is mobile check deposit available on my account?",
            tests=(
                _t("Is mobile check deposit unavailable on my account?", False, NEGATION, "available->unavailable"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="ATM fee reimbursement (negation)",
            cached_query="Are ATM fees reimbursed on premium accounts?",
            tests=(
                _t("Are ATM fees NOT reimbursed on premium accounts?", False, NEGATION, "explicit negation"),
            ),
        ),
        # ---------------- Destructive / state-mutating actions ----------------
        AdversarialCacheScenario(
            scenario="Cancel Zelle payment (destructive action)",
            cached_query="Cancel my pending Zelle payment.",
            tests=(
                _t("Stop the Zelle transfer I just sent.", False, ACTION, "paraphrased action"),
                _t("Please cancel my pending wire transfer.", False, ACTION, "different action target"),
                _t("How do I cancel a Zelle payment?", False, ACTION, "how-to vs action"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Lock stolen debit card (security action)",
            cached_query="Lock my debit card immediately, it was stolen.",
            tests=(
                _t("Freeze my debit card - I think it's compromised.", False, ACTION, "paraphrased action"),
                _t("Block my debit card right now.", False, ACTION, "paraphrased action"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Close account (destructive action)",
            cached_query="Close my savings account effective today.",
            tests=(
                _t("Shut down my savings account now.", False, ACTION, "paraphrased action"),
                _t("Close my checking account today.", False, ACTION, "different account target"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Move funds (state-mutating action + amount)",
            cached_query="Transfer $5,000 from savings to checking.",
            tests=(
                _t("Move $5,000 from savings into checking.", False, ACTION, "paraphrased action"),
                _t("Transfer $500 from savings to checking.", False, ACTION, "amount swap + action"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="File charge dispute (action + amount)",
            cached_query="Dispute the $89.99 charge from my last statement.",
            tests=(
                _t("File a dispute for that $89.99 charge.", False, ACTION, "paraphrased action"),
                _t("Open a dispute on the $129.99 charge.", False, ACTION, "amount swap + action"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Raise credit limit (action)",
            cached_query="Increase my credit card limit to $10,000.",
            tests=(
                _t("Raise my credit limit to $10,000.", False, ACTION, "paraphrased action"),
            ),
        ),
        # ---------------- Freshness-sensitive ----------------
        AdversarialCacheScenario(
            scenario="Live checking balance (freshness)",
            cached_query="What is my checking account balance right now?",
            tests=(
                _t("What's my current checking balance?", False, FRESHNESS, "value changes"),
                _t("What was my checking balance at close yesterday?", False, FRESHNESS, "temporal shift"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Live mortgage rate (freshness)",
            cached_query="What's today's 30-year fixed mortgage rate?",
            tests=(
                _t("What is the current 30-year fixed mortgage rate?", False, FRESHNESS, "live rate"),
                _t("What's today's 15-year fixed mortgage rate?", False, FRESHNESS, "term swap + live rate"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Live savings APY (freshness)",
            cached_query="What's the current APY on the high-yield savings account?",
            tests=(
                _t("What APY is the high-yield savings paying right now?", False, FRESHNESS, "live rate"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Available credit now (freshness)",
            cached_query="How much available credit do I have right now?",
            tests=(
                _t("What's my current available credit?", False, FRESHNESS, "value changes"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Live FX rate (freshness / currency)",
            cached_query="What's the USD to EUR exchange rate today?",
            tests=(
                _t("What's the current USD to EUR rate?", False, FRESHNESS, "live rate"),
                _t("What's the USD to GBP rate today?", False, FRESHNESS, "currency swap + live rate"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Pending transactions now (freshness)",
            cached_query="What are my pending transactions right now?",
            tests=(
                _t("Show my current pending transactions.", False, FRESHNESS, "value changes"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Credit score today (freshness)",
            cached_query="What's my credit score as of today?",
            tests=(
                _t("What's my current credit score?", False, FRESHNESS, "value changes"),
            ),
        ),
        # ---------------- Principal / account-holder swaps ----------------
        AdversarialCacheScenario(
            scenario="Account number by holder (paraphrase / principal swap)",
            cached_query="What's the account number on file for customer Jack Miller?",
            tests=(
                _t("Which account number do we have on file for Jack Miller?", True, PARAPHRASE, "same principal, rephrase"),
                _t("What's the account number on file for customer Jill Carter?", False, PRINCIPAL_SWAP, "different holder"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Mailing address by holder (principal swap)",
            cached_query="What's the mailing address on file for John Smith?",
            tests=(
                _t("What's the mailing address on file for Jane Smith?", False, PRINCIPAL_SWAP, "different holder"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Recent transactions by holder (principal swap)",
            cached_query="List the last 5 transactions for Maria Lopez's account.",
            tests=(
                _t("List the last 5 transactions for Maria Gomez's account.", False, PRINCIPAL_SWAP, "different holder, shared first name"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Phone on file by holder (principal swap)",
            cached_query="What's the phone number on file for account holder Robert Chen?",
            tests=(
                _t("What's the phone number on file for account holder Rachel Chen?", False, PRINCIPAL_SWAP, "different holder, shared surname"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Credit limit by holder (principal swap)",
            cached_query="What's the credit limit on David Park's card?",
            tests=(
                _t("What's the credit limit on Diana Park's card?", False, PRINCIPAL_SWAP, "different holder, shared surname"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Account open date by holder (principal swap)",
            cached_query="When did customer Alan Wright open his account?",
            tests=(
                _t("When did customer Alice Wright open her account?", False, PRINCIPAL_SWAP, "different holder, shared surname"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Email by member (same id, different person) (principal swap)",
            cached_query="What's the email on file for member #M-2201, Tom Reed?",
            tests=(
                _t("What's the email on file for member #M-2201, Tim Reed?", False, PRINCIPAL_SWAP, "same member id, different person"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="SSN on file by holder (principal swap)",
            cached_query="What's the SSN on file for customer Greg Hall?",
            tests=(
                _t("What's the SSN on file for customer Gary Hall?", False, PRINCIPAL_SWAP, "different holder, shared surname"),
            ),
        ),
        # ---------------- Product-swap top-ups ----------------
        AdversarialCacheScenario(
            scenario="Credit card grace period (paraphrase / product)",
            cached_query="What's the grace period on my credit card payment?",
            tests=(
                _t("How many days is the grace period before credit card interest accrues?", True, PARAPHRASE, "rephrase"),
                _t("What's the grace period on my personal loan payment?", False, PRODUCT_SWAP, "credit card->loan"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Documents to open account (paraphrase / product)",
            cached_query="What documents do I need to open a checking account?",
            tests=(
                _t("What's required to open a checking account?", True, PARAPHRASE, "rephrase"),
                _t("What documents do I need to open a mortgage?", False, PRODUCT_SWAP, "checking->mortgage"),
            ),
        ),
    ]


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
@dataclass
class Confusion:
    tp: int = 0  # should-reuse and reused (good)
    fn: int = 0  # should-reuse but missed (lost hit)
    fp: int = 0  # should-NOT-reuse but reused (WRONG ANSWER)
    tn: int = 0  # should-NOT-reuse and not reused (good)

    def add(self, expected_hit: bool, actual_hit: bool) -> None:
        if expected_hit and actual_hit:
            self.tp += 1
        elif expected_hit and not actual_hit:
            self.fn += 1
        elif not expected_hit and actual_hit:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def tpr(self) -> float | None:
        pos = self.tp + self.fn
        return self.tp / pos if pos else None

    @property
    def fpr(self) -> float | None:
        neg = self.fp + self.tn
        return self.fp / neg if neg else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp, "fn": self.fn, "fp": self.fp, "tn": self.tn,
            "recall_tpr": None if self.tpr is None else round(self.tpr, 4),
            "false_reuse_fpr": None if self.fpr is None else round(self.fpr, 4),
        }


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


SUITES = {
    "core32": build_banking_adversarial_scenarios,
    "full100": build_banking_adversarial_scenarios_100,
}


def run_banking_comparison(
    *,
    design_threshold: float = 0.86,
    baseline_threshold: float = 0.85,
    openai_model: str | None = None,
    suite: str = "core32",
) -> dict[str, Any]:
    """Run the design pipeline and a cosine-only baseline on the same banking traps."""

    model = openai_model or os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {sorted(SUITES)}")
    scenarios = SUITES[suite]()

    # ---- Design system: full pipeline via the shared adversarial runner ----
    design_report = run_adversarial_cache_eval(
        semantic_threshold=design_threshold,
        openai_model=model,
        scenarios=scenarios,
    )

    # ---- Baseline system: cosine-only, same embeddings ----
    from gatecache.embeddings.backends import make_openai_embedder

    embedder = make_openai_embedder(model=model)
    _cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in _cache:
            _cache[text] = embedder(text)
        return _cache[text]

    baseline_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        seed_vec = embed(scenario.cached_query)
        for test in scenario.tests:
            sim = _cosine(seed_vec, embed(test.query))
            actual_hit = sim >= baseline_threshold
            baseline_rows.append(
                {
                    "scenario": scenario.scenario,
                    "cached_query": scenario.cached_query,
                    "query": test.query,
                    "category": _category_of(test.note),
                    "expected_cache_hit": test.cache_hit,
                    "actual_cache_hit": actual_hit,
                    "passed": actual_hit is test.cache_hit,
                    "cosine": round(sim, 4),
                    "note": test.note,
                }
            )

    # ---- Aggregate confusion matrices, overall + per category ----
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        overall = Confusion()
        by_cat: dict[str, Confusion] = {}
        for r in rows:
            cat = r["category"]
            overall.add(r["expected_cache_hit"], r["actual_cache_hit"])
            by_cat.setdefault(cat, Confusion()).add(r["expected_cache_hit"], r["actual_cache_hit"])
        return {
            "overall": overall.to_dict(),
            "by_category": {k: v.to_dict() for k, v in sorted(by_cat.items())},
        }

    design_rows = [
        {
            "scenario": row.scenario,
            "cached_query": row.cached_query,
            "query": row.query,
            "category": _category_of(row.note),
            "expected_cache_hit": row.expected_cache_hit,
            "actual_cache_hit": row.actual_cache_hit,
            "passed": row.passed,
            "source": row.source,
            "routing_label": row.routing_label,
            "routing_confidence": row.routing_confidence,
            "top_candidate_similarity": row.top_candidate_similarity,
            "neighbor_judge_invoked": row.neighbor_judge_invoked,
            "reject_reason": row.semantic_post_ann_reject_reason,
            "note": row.note,
        }
        for row in design_report.rows
    ]

    return {
        "name": "banking_adversarial_eval",
        "suite": suite,
        "model": model,
        "design_threshold": design_threshold,
        "baseline_threshold": baseline_threshold,
        "total_candidates": len(design_rows),
        "design": {"metrics": aggregate(design_rows), "rows": design_rows},
        "baseline_cosine_only": {"metrics": aggregate(baseline_rows), "rows": baseline_rows},
    }


def _fmt(conf: dict[str, Any]) -> str:
    tpr = conf["recall_tpr"]
    fpr = conf["false_reuse_fpr"]
    tpr_s = "  n/a" if tpr is None else f"{tpr:5.2f}"
    fpr_s = "  n/a" if fpr is None else f"{fpr:5.2f}"
    return f"TPR={tpr_s}  FPR(false-reuse)={fpr_s}  [tp={conf['tp']} fn={conf['fn']} fp={conf['fp']} tn={conf['tn']}]"


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Banking adversarial cache eval: design vs cosine-only baseline")
    parser.add_argument("--design-threshold", type=float, default=0.86)
    parser.add_argument("--baseline-threshold", type=float, default=0.85)
    parser.add_argument("--suite", choices=sorted(SUITES), default="core32")
    parser.add_argument("--report-json", type=str, default=None)
    args = parser.parse_args()

    try:
        from gatecache.cli import _load_dotenv_files

        _load_dotenv_files()
    except Exception:
        pass

    report = run_banking_comparison(
        design_threshold=args.design_threshold,
        baseline_threshold=args.baseline_threshold,
        suite=args.suite,
    )

    print(f"\n=== Banking adversarial cache eval [{report['suite']}] ({report['total_candidates']} candidates) ===")
    print(f"model={report['model']}  design_thresh={report['design_threshold']}  baseline_thresh={report['baseline_threshold']}\n")
    print("DESIGN (full pipeline):")
    print("  overall  ", _fmt(report["design"]["metrics"]["overall"]))
    for cat, conf in report["design"]["metrics"]["by_category"].items():
        print(f"    {cat:<16}", _fmt(conf))
    print("\nBASELINE (cosine-only, LiteLLM/Redis-style):")
    print("  overall  ", _fmt(report["baseline_cosine_only"]["metrics"]["overall"]))
    for cat, conf in report["baseline_cosine_only"]["metrics"]["by_category"].items():
        print(f"    {cat:<16}", _fmt(conf))

    if args.report_json:
        with open(args.report_json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote report to {args.report_json}")


if __name__ == "__main__":
    main()
