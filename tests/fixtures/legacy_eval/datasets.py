from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class EvalCase:
    session_id: str
    user_message: str
    expected_hit_allowed: bool
    expected_bypass: bool
    expected_answer_class: str
    expected_scope: str


def build_dataset() -> list[EvalCase]:
    return build_comprehensive_dataset()


def _append_session(
    cases: list[EvalCase],
    session_id: str,
    prompts: list[tuple[str, bool, bool, str, str]],
) -> None:
    for user_message, expected_hit_allowed, expected_bypass, answer_class, scope in prompts:
        cases.append(EvalCase(session_id, user_message, expected_hit_allowed, expected_bypass, answer_class, scope))


def build_retrieval_ablation_dataset() -> list[EvalCase]:
    cases: list[EvalCase] = []

    _append_session(
        cases,
        "ra_sales_1",
        [
            ("List top 5 iphone sales in europe last 7 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("list top 5 iphone sales in europe last 7 days", True, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("Show me the top 5 iphone sales in europe for the last seven days", True, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("Show me the top five iphone sales across europe in the past week", True, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("List top 10 iphone sales in europe last 7 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("that's wrong", False, True, "DYNAMIC_FACT", "SLOW_MOVING"),
        ],
    )
    _append_session(
        cases,
        "ra_sales_2",
        [
            ("List top 5 revenue in apac today", False, False, "DYNAMIC_FACT", "REALTIME"),
            ("show top 5 revenue in apac today", True, False, "DYNAMIC_FACT", "REALTIME"),
            ("List top 5 revenue in apac now", True, False, "DYNAMIC_FACT", "REALTIME"),
            ("List top 5 revenue in emea today", False, False, "DYNAMIC_FACT", "REALTIME"),
            ("are you sure", False, True, "DYNAMIC_FACT", "REALTIME"),
        ],
    )
    _append_session(
        cases,
        "ra_docs_1",
        [
            ("Explain what semantic caching is", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("What is semantic caching?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Can you explain semantic caching?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Explain customer onboarding", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_docs_2",
        [
            ("Summarize docs for API auth", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("summarize documentation for api auth", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("give me a summary of the api auth docs", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("How to configure support webhook", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_lookup_1",
        [
            ("Lookup order #A123 status", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("find order A123 status", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("search for order a123 status", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Lookup order #B456 status", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_lookup_2",
        [
            ("Lookup customer #C777 profile", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("find customer c777 profile", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("search customer C777 profile", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Lookup customer #C778 profile", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_compare_1",
        [
            ("Compare iphone vs ipad in europe last 30 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("compare iphone and ipad in europe over the last 30 days", True, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("Compare iphone vs ipad in europe last 7 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
        ],
    )
    _append_session(
        cases,
        "ra_location_1",
        [
            ("closest brewpubs to me", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("closest brewpubs to me", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_structured_1",
        [
            (
                "show me all adult pants size 32 inch waist, blue, cotton and made in america",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "show me all adult pants size 34 inch waist, blue, cotton and made in america",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
        ],
    )
    _append_session(
        cases,
        "ra_travel_1",
        [
            (
                "find me nonstop flights from sfo to tokyo in november with premium economy and one checked bag",
                False,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
            (
                "find me nonstop flights from sfo to tokyo in november with premium economy and two checked bag",
                False,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
        ],
    )
    _append_session(
        cases,
        "ra_mutation_1",
        [
            ("List top 5 widget sales in us today", False, False, "DYNAMIC_FACT", "REALTIME"),
            ("Now do same for europe", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Now do same for europe", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Also for apac", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Also for apac", True, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_mutation_2",
        [
            ("List top 5 iphone sales in europe last 7 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("Now do same for us", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Now do same for us", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Also for apac", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Also for apac", True, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_sales_3",
        [
            ("List top 5 macbook revenue in us today", False, False, "DYNAMIC_FACT", "REALTIME"),
            ("list top 5 macbook revenue in us today", True, False, "DYNAMIC_FACT", "REALTIME"),
            ("show top 5 macbook revenue in us today", True, False, "DYNAMIC_FACT", "REALTIME"),
            ("List top 5 macbook revenue in us yesterday", False, False, "DYNAMIC_FACT", "REALTIME"),
            ("recheck that", False, True, "DYNAMIC_FACT", "REALTIME"),
        ],
    )
    _append_session(
        cases,
        "ra_sales_4",
        [
            ("Compare widget vs macbook in emea last 30 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("compare widget and macbook in emea over the last 30 days", True, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("Compare widget vs macbook in emea last 7 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("Compare widget vs iphone in emea last 30 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
        ],
    )
    _append_session(
        cases,
        "ra_docs_3",
        [
            ("How to configure support webhook", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("how do i configure a support webhook", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("explain how to configure support webhook", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("How to configure sales webhook", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_docs_4",
        [
            ("Summarize docs for customer onboarding", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("summarize documentation for customer onboarding", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("give me a summary of customer onboarding docs", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Summarize docs for API auth", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_lookup_3",
        [
            ("Lookup order #Z999 status", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("find order z999 status", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("search order Z999 status", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Lookup order #Z998 status", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_lookup_4",
        [
            ("Lookup customer #C901 profile", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("find customer c901 profile", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("search for customer C901 profile", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Lookup customer #C902 profile", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_mutation_3",
        [
            ("List top 5 revenue in apac today", False, False, "DYNAMIC_FACT", "REALTIME"),
            ("Now do same for emea", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Now do same for emea", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Also for europe", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Also for europe", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("that's wrong", False, True, "DYNAMIC_FACT", "REALTIME"),
        ],
    )
    _append_session(
        cases,
        "ra_mutation_4",
        [
            ("Compare iphone vs ipad in europe last 30 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("Now do same for us", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Now do same for us", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Also for apac", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Also for apac", True, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ra_branch_1",
        [
            ("List top 5 iphone sales in europe last 7 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"),
            ("Now do same for us", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("go back to europe", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("the first one", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    cases.extend(build_support_semantic_cases())
    cases.extend(build_bitext_inspired_cases())
    return cases


def build_support_semantic_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    _append_session(
        cases,
        "support_connectivity_1",
        [
            (
                "I'm having trouble connecting to my RDS instance from my local machine. I keep getting a timeout error even though the status says 'Available'.",
                False,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
            (
                "Why is my connection to the managed SQL database timing out? The console says it's running but I can't reach it from home.",
                True,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
            (
                "The database says available, but my office IP can connect while my home IP still times out. Is this still just a firewall issue?",
                False,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
        ],
    )
    _append_session(
        cases,
        "support_scaling_1",
        [
            (
                "Our production database is hitting 95% CPU utilization. Should we upgrade the instance class or is there a way to optimize the current one?",
                False,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
            (
                "The DB server is pegged at max CPU. Is it time to scale up the hardware, or should I look for slow queries first?",
                True,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
            (
                "CPU is fine now, but memory is saturating instead. Should we still upgrade the instance type?",
                False,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
        ],
    )
    _append_session(
        cases,
        "support_backup_1",
        [
            (
                "I accidentally deleted a table in my staging environment. Can I restore just that specific table from last night's snapshot?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "Is it possible to do a point-in-time recovery for a single table I dropped by mistake? I have daily backups enabled.",
                True,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "Can I restore the whole staging instance from yesterday's backup instead of just one table?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
        ],
    )
    _append_session(
        cases,
        "support_storage_1",
        [
            (
                "My database is in a 'Storage Full' state and I can't run any write operations. How do I fix this quickly?",
                False,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
            (
                "We ran out of disk space on the cloud DB and now it's read-only. What are the steps to add more GBs?",
                True,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
            (
                "The disk is full, but I want to shrink the storage allocation after cleanup. Is that allowed?",
                False,
                False,
                "DYNAMIC_FACT",
                "REALTIME",
            ),
        ],
    )
    return cases


def build_bitext_inspired_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    _append_session(
        cases,
        "bitext_shipping_1",
        [
            ("I need to change the shipping address for my order before it goes out.", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Can I update the delivery address on an order that hasn't shipped yet?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("The package already shipped. Can I reroute it to another city instead?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "bitext_refund_1",
        [
            ("What's your refund policy for an online order that arrived damaged?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Can you explain the return and refund policy if an order comes damaged?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("How do I track a refund after it was already approved?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "bitext_payment_1",
        [
            ("My credit card keeps getting declined at checkout. What should I do?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("I'm having a payment issue at checkout because my card won't go through. How can I fix it?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("What payment methods do you accept for international orders?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "bitext_account_1",
        [
            ("How do I delete my account permanently?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Can you tell me the steps to permanently remove my account?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("I just want to switch from a personal account to a business account.", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "bitext_order_1",
        [
            ("Can I cancel an online order I placed this morning?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Is it still possible to cancel an order I submitted earlier today?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Can I change the items in that order instead of cancelling it?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "bitext_invoice_1",
        [
            ("Where can I download the invoice for order A123?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("How do I get the invoice for my order A123?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("How can I check whether invoice B456 was already issued?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    return cases


def build_ibm_inferred_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    _append_session(
        cases,
        "ibm_safe_room_1",
        [
            ("What are the sheltered rooms designated for use?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("What are safe rooms meant to protect people from?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Is it the same for earthquakes?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ibm_groundwater_1",
        [
            ("What is ground water contamination?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("What does groundwater contamination mean?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Can it be clean up?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("What is it?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ibm_bank_account_1",
        [
            ("Freelancer: Should I start a second bank account?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("As a freelancer, is it a good idea to open a second bank account for business?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Is it free to open a second bank account?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ibm_webchat_1",
        [
            ("Is it worth having a web chat widget on my website?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Would adding a web chat widget to my site be worthwhile?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("how to add web chat widget?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Can I extend the web chat? if so, how can I do that?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ibm_nfl_1",
        [
            ("How many teams are in the NFL?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("How many NFL teams are there?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("How many teams are in the NFL playoffs?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ibm_wacc_1",
        [
            ("What is WACC?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Can you explain what weighted average cost of capital means?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("What is the ideal formula to evaluate a company?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ibm_day_trading_1",
        [
            ("ok. I'll do day trading in an IRA account. Can I start with a 50K deposit?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("If I day trade inside an IRA, is starting with a $50k deposit reasonable?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("Is it still advisable to do day trading in an IRA account, with just $5500?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    _append_session(
        cases,
        "ibm_bicycles_1",
        [
            ("What about bicycles?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("How about bikes?", True, False, "STATIC_KNOWLEDGE", "STATIC"),
            ("What about mediation?", False, False, "STATIC_KNOWLEDGE", "STATIC"),
        ],
    )
    return cases


def build_general_robustness_dataset() -> list[EvalCase]:
    return build_ibm_inferred_cases()


def build_novel_domain_dataset() -> list[EvalCase]:
    """Novel finance/healthcare eval set grounded in external conversation data.

    Prompt sources:
    - finance prompts adapted from Fin-Ally (`data.csv`)
    - healthcare prompts adapted from ChatDoctor (`chatdoctor5k.json`)

    We still add explicit expected-hit / expected-miss labels, because semantic
    cache evaluation needs known reuse outcomes rather than open-ended threads.
    """

    cases: list[EvalCase] = []

    _append_session(
        cases,
        "novel_finance_credit_payoff_1",
        [
            (
                "What's the best way to use a credit card? Should I pay it off in full every month or make steady payments?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "Is it better to pay my credit card balance in full every month, or should I spread the payments out over time?",
                True,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "Now tell me, when should I use a debit card instead of a credit card for everyday expenses?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "That doesn't sound right.",
                False,
                True,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
        ],
    )

    _append_session(
        cases,
        "novel_finance_tuition_1",
        [
            (
                "Does paying off my school tuition with a credit card provide any real benefit compared to using a debit card? My job deposits the tuition amount directly into my bank account, and I have the full amount available.",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "If I already have the tuition money in my bank account, is there any real upside to paying the school with a credit card instead of my debit card?",
                True,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "What if I put the tuition on my credit card and saved the money they give me, then split it up monthly? I got a new card with a 15-month 0% APR.",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "No, I want to avoid carrying a balance entirely. Does that change your recommendation?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
        ],
    )

    _append_session(
        cases,
        "novel_finance_savings_1",
        [
            (
                "What’s the difference between a savings account and a checking account?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "Can you explain how a savings account differs from a checking account?",
                True,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "Also wanted to know if I should keep all my money in one place?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "No, I mean only for day-to-day spending money.",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
        ],
    )

    _append_session(
        cases,
        "novel_healthcare_panic_1",
        [
            (
                "Doctor, I have been experiencing sudden and frequent panic attacks. I don't know what to do.",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "I've started having sudden panic attacks over and over again. What should I do next?",
                True,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "I'm having sudden panic attacks and chest pain after starting a new medication. What should I do?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "No, these are happening every day now.",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
        ],
    )

    _append_session(
        cases,
        "novel_healthcare_sinusitis_1",
        [
            (
                "Doctor, I have been having severe headaches and nasal congestion for the past few days. I also have a fever and my face feels very tender. I think I might have acute sinusitis.",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "I've had bad headaches, congestion, fever, and facial tenderness for days. Could this be acute sinusitis?",
                True,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "I've had the same headache and congestion, but now I'm also having blurry vision. Could this still be sinusitis?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "That doesn't seem to match what I'm feeling.",
                False,
                True,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
        ],
    )

    _append_session(
        cases,
        "novel_healthcare_swallowed_object_1",
        [
            (
                "Doctor, I think I have swallowed a small object, and it's stuck somewhere in my digestive system. What should I do?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "I may have swallowed a small object and it feels like it's stuck in my digestive tract. What should I do now?",
                True,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "I think I swallowed a button battery and it feels stuck. What should I do?",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
            (
                "No, it doesn't just feel uncomfortable. It hurts when I swallow.",
                False,
                False,
                "STATIC_KNOWLEDGE",
                "STATIC",
            ),
        ],
    )

    return cases


def build_domain_specific_dataset() -> list[EvalCase]:
    return build_retrieval_ablation_dataset()


def build_eval_dataset(name: str) -> list[EvalCase]:
    key = name.strip().lower()
    if key in {"retrieval", "domain", "domain_specific"}:
        return build_domain_specific_dataset()
    if key in {"novel", "novel_domain", "healthcare_finance"}:
        return build_novel_domain_dataset()
    if key in {"general", "robustness", "general_robustness"}:
        return build_general_robustness_dataset()
    if key in {"fixed", "comprehensive"}:
        return build_comprehensive_dataset()
    raise ValueError(f"Unknown dataset name: {name}")


def build_comprehensive_dataset() -> list[EvalCase]:
    cases = build_domain_specific_dataset()
    cases.extend(build_general_robustness_dataset())
    cases.extend(build_fixed_dataset())
    return cases


def build_fixed_dataset() -> list[EvalCase]:
    cases: list[EvalCase] = []
    bases = [
        ("List top 5 iphone sales in europe last 7 days", "DYNAMIC_FACT", "SLOW_MOVING"),
        ("Explain what semantic caching is", "STATIC_KNOWLEDGE", "STATIC"),
        ("Lookup order #A123 status", "STATIC_KNOWLEDGE", "STATIC"),
        ("List top 5 widget sales in us today", "DYNAMIC_FACT", "REALTIME"),
        ("Summarize docs for API auth", "STATIC_KNOWLEDGE", "STATIC"),
        ("Compare iphone vs ipad in europe last 30 days", "DYNAMIC_FACT", "SLOW_MOVING"),
        ("How to configure support webhook", "STATIC_KNOWLEDGE", "STATIC"),
        ("List top 5 revenue in apac today", "DYNAMIC_FACT", "REALTIME"),
        ("Explain customer onboarding", "STATIC_KNOWLEDGE", "STATIC"),
        ("Lookup customer #C777 profile", "STATIC_KNOWLEDGE", "STATIC"),
    ]

    for i, (prompt, answer_class, scope) in enumerate(bases, 1):
        cases.append(EvalCase(f"s{i}", prompt, False, False, answer_class, scope))
        cases.append(
            EvalCase(
                f"s{i}",
                prompt.replace("List", "list").replace("Explain", "explain").replace("Lookup", "lookup"),
                True,
                False,
                answer_class,
                scope,
            )
        )

    for i in range(5):
        cases.append(EvalCase(f"t{i}", "List top 10 iphone sales in europe last 7 days", False, False, "DYNAMIC_FACT", "SLOW_MOVING"))

    for i, d in enumerate(["that's wrong", "are you sure", "recheck", "incorrect", "wrong"]):
        cases.append(EvalCase(f"d{i}", f"{d} about iphone sales today", False, True, "DYNAMIC_FACT", "REALTIME"))

    for i in range(10):
        cases.append(EvalCase(f"m{i}", "Now do same for europe", False, False, "STATIC_KNOWLEDGE", "STATIC"))

    return cases[:40]


def build_synthetic_dataset(num_cases: int = 500, seed: int = 7) -> list[EvalCase]:
    """Generate deterministic multi-turn synthetic conversational workload."""
    rng = random.Random(seed)

    products = ["iphone", "ipad", "widget", "macbook"]
    regions = ["europe", "us", "apac", "emea"]
    timeframes = [("today", "DYNAMIC_FACT", "REALTIME"), ("last 7 days", "DYNAMIC_FACT", "SLOW_MOVING")]
    actions = ["sales", "revenue"]

    cases: list[EvalCase] = []
    session_count = max(1, num_cases // 5)

    for i in range(session_count):
        session_id = f"sx{i}"
        product = products[i % len(products)]
        region = regions[i % len(regions)]
        tf_text, ac, scope = timeframes[i % len(timeframes)]
        metric = actions[i % len(actions)]

        # Turn 1: cache seed (miss expected)
        prompt = f"List top 5 {product} {metric} in {region} {tf_text}"
        cases.append(EvalCase(session_id, prompt, False, False, ac, scope))

        # Turn 2: paraphrase (hit expected)
        paraphrase = f"list top 5 {product} {metric} in {region} {tf_text}"
        cases.append(EvalCase(session_id, paraphrase, True, False, ac, scope))

        # Turn 3: strict mismatch (top 10 should not hit top 5)
        mismatch = f"List top 10 {product} {metric} in {region} {tf_text}"
        cases.append(EvalCase(session_id, mismatch, False, False, ac, scope))

        # Turn 4: dispute/correction should bypass
        dispute_variants = ["that's wrong", "are you sure", "recheck that", "incorrect", "wrong"]
        d = dispute_variants[i % len(dispute_variants)]
        cases.append(EvalCase(session_id, f"{d} about {product} {metric} {tf_text}", False, True, ac, scope))

        # Turn 5: mutation on region should generally miss
        next_region = regions[(i + 1) % len(regions)]
        mutation_prefix = "Now do same for" if rng.randint(0, 1) == 0 else "Also for"
        mutation = f"{mutation_prefix} {next_region}"
        cases.append(EvalCase(session_id, mutation, False, False, "STATIC_KNOWLEDGE", "STATIC"))

    return cases[:num_cases]
