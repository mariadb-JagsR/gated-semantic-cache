from __future__ import annotations

from dataclasses import dataclass

from gated_semantic_cache.routing.labels import RoutingLabel


@dataclass(frozen=True, slots=True)
class RoutingExample:
    query: str
    label: RoutingLabel
    slice_id: str
    source: str
    notes: str = ""
    thread_scope_present: bool = False
    namespace_policy: str = "default"


def build_routing_dataset() -> list[RoutingExample]:
    examples: list[RoutingExample] = []
    examples.extend(_semantic_ok_examples())
    examples.extend(_skip_cache_examples())
    examples.extend(_exact_only_examples())
    examples.extend(_thread_scoped_examples())
    return examples


def _semantic_ok_examples() -> list[RoutingExample]:
    rows = [
        ("Explain what semantic caching is", "A", "legacy_eval", "faq style reusable question"),
        ("What is semantic caching?", "A", "legacy_eval", "paraphrase of reusable question"),
        ("Can you explain semantic caching?", "F", "legacy_eval", "paraphrase slice"),
        ("How does API auth work?", "A", "legacy_eval", "stable docs question"),
        ("Does the platform support row level security?", "A", "manual", "product capability question"),
        ("How do I configure a support webhook?", "A", "legacy_eval", "stable documentation lookup"),
        ("Summarize docs for customer onboarding", "A", "legacy_eval", "documentation summary"),
        ("Does the cache support namespace isolation?", "A", "manual", "product capability"),
        ("What are the limits of exact cache reuse?", "A", "manual", "generic architecture question"),
        ("Explain the difference between exact cache and semantic cache", "F", "manual", "paraphrase-safe architecture question"),
        ("How does cache invalidation work for docs answers?", "A", "manual", "stable how-it-works question"),
        ("Can the system reuse stable FAQ answers?", "F", "manual", "safe capability paraphrase"),
        ("How do semantic cache thresholds get tuned?", "A", "manual", "stable operational question"),
        ("What does thread scoped reuse mean?", "A", "manual", "reusable conceptual explanation"),
        (
            "What's the best way to use a credit card? Should I pay it off in full every month or make steady payments?",
            "A",
            "legacy_eval",
            "legacy novel-domain informational query",
        ),
        (
            "Is it better to pay my credit card balance in full every month, or should I spread the payments out over time?",
            "F",
            "legacy_eval",
            "legacy novel-domain paraphrase",
        ),
        (
            "What’s the difference between a savings account and a checking account?",
            "A",
            "legacy_eval",
            "legacy novel-domain informational query",
        ),
        (
            "Can you explain how a savings account differs from a checking account?",
            "F",
            "legacy_eval",
            "legacy novel-domain paraphrase",
        ),
        (
            "What payment methods do you accept for international orders?",
            "A",
            "legacy_eval",
            "legacy policy question",
        ),
        (
            "What's your refund policy for an online order that arrived damaged?",
            "A",
            "legacy_eval",
            "legacy policy question",
        ),
        (
            "Can you explain the return and refund policy if an order comes damaged?",
            "F",
            "legacy_eval",
            "legacy policy paraphrase",
        ),
        (
            "How do I delete my account permanently?",
            "A",
            "legacy_eval",
            "legacy docs-style policy question, not an execution request",
        ),
        (
            "Can you tell me the steps to permanently remove my account?",
            "F",
            "legacy_eval",
            "legacy docs-style paraphrase",
        ),
        (
            "Can I cancel an online order I placed this morning?",
            "A",
            "legacy_eval",
            "legacy policy/explanation question",
        ),
        (
            "Is it still possible to cancel an order I submitted earlier today?",
            "F",
            "legacy_eval",
            "legacy policy paraphrase",
        ),
        (
            "Where can I download the invoice for order A123?",
            "A",
            "legacy_eval",
            "policy/how-to question with anchor mention but informational intent",
        ),
        (
            "How do I get the invoice for my order A123?",
            "F",
            "legacy_eval",
            "legacy anchored policy paraphrase",
        ),
        (
            "How does namespace isolation affect cache reuse?",
            "A",
            "manual",
            "stable architecture question",
        ),
        (
            "Does the cache work for reusable documentation questions?",
            "F",
            "manual",
            "paraphrase-safe product capability",
        ),
        (
            "What is the safest way to rotate api keys without downtime?",
            "A",
            "manual",
            "explanatory question about an action, not a direct execution request",
        ),
        (
            "How do I revoke an access token in the product?",
            "A",
            "manual",
            "documentation-style how-to question",
        ),
        (
            "What does the cancellation policy say for orders that already shipped?",
            "A",
            "manual",
            "policy question that should remain reusable",
        ),
        (
            "Explain how order rerouting works after shipment",
            "A",
            "manual",
            "policy/explanation question near a risky action topic",
        ),
        (
            "List the symptoms of a pulmonary embolism.",
            "A",
            "queries_txt",
            "general medical fact question (not thread follow-up)",
        ),
        (
            "What are the common side effects of Lisinopril?",
            "M",
            "queries_txt",
            "drug-specific adverse effect question (pair with ACE-class paraphrase)",
        ),
        (
            "Tell me about the adverse reactions associated with ACE inhibitors like Lisinopril.",
            "M",
            "queries_txt",
            "ACE inhibitor class + drug name — same SEMANTIC_OK intent as drug-specific sibling",
        ),
        (
            "Does Metformin cause stomach issues?",
            "M",
            "queries_txt",
            "Metformin GI question (pair with paraphrase)",
        ),
        (
            "Is gastrointestinal distress a frequent complication of taking Metformin?",
            "M",
            "queries_txt",
            "Metformin GI paraphrase — align with sibling",
        ),
        (
            "What are the international wire transfer fees for a business account sending USD to a bank in Japan?",
            "A",
            "manual",
            "stable banking fee policy question",
        ),
        (
            "international wire transfer fees for a business account sending USD",
            "F",
            "manual",
            "short paraphrase of stable banking fee policy",
        ),
        (
            "How much does a business USD wire transfer to Japan cost?",
            "F",
            "manual",
            "banking fee paraphrase",
        ),
        (
            "What are the limits for international wire transfers from a business account?",
            "A",
            "manual",
            "stable banking limits policy",
        ),
        (
            "Explain the refund fees for marketplace sellers this month.",
            "A",
            "manual",
            "TTL-cacheable policy question with a time window",
        ),
        (
            "What is the pricing for storage overage this month?",
            "A",
            "manual",
            "TTL-cacheable pricing documentation question",
        ),
        (
            "dimensions for Samsung washer D1234",
            "A",
            "manual",
            "short standalone product documentation request with a protected token",
        ),
        (
            "limits for db.r6g.large",
            "A",
            "manual",
            "short standalone infrastructure limits request with a code-like token",
        ),
        (
            "docs for max_connections",
            "A",
            "manual",
            "short standalone documentation request with a code-like token",
        ),
        (
            "What were the documented API rate limits last month?",
            "A",
            "manual",
            "TTL-cacheable documentation question with historical window",
        ),
        (
            "How does payroll tax withholding work for contractors?",
            "A",
            "manual",
            "general finance explanation",
        ),
        (
            "What are the baggage fees for international economy flights?",
            "A",
            "manual",
            "stable travel policy question",
        ),
        (
            "Explain the warranty coverage for refurbished laptops.",
            "A",
            "manual",
            "stable commerce policy question",
        ),
        (
            "What are the eligibility requirements for a small business loan?",
            "A",
            "manual",
            "general lending policy question",
        ),
        (
            "How do Kubernetes pod disruption budgets work?",
            "A",
            "manual",
            "stable technical documentation question",
        ),
        (
            "What is the difference between OAuth scopes and API roles?",
            "A",
            "manual",
            "stable technical concept comparison",
        ),
        (
            "How do I configure automated backups for a managed database?",
            "A",
            "manual",
            "stable database how-to",
        ),
        (
            "What are the data retention rules for audit logs this month?",
            "A",
            "manual",
            "TTL-cacheable compliance policy question",
        ),
        (
            "Show me pants with waist 32, length 32, cotton, must be made in us, doesn't fade, black",
            "A",
            "manual",
            "cacheable retail product search with structured filters",
        ),
        (
            "Find black cotton pants with a 32 inch waist and 32 inch length made in the US",
            "F",
            "manual",
            "retail product filter paraphrase",
        ),
        (
            "Show adult jeans size 34 waist blue denim made in America",
            "A",
            "manual",
            "cacheable retail product search with color/material filters",
        ),
        (
            "Find wireless noise cancelling headphones under 200 USD",
            "A",
            "manual",
            "cacheable product search with price bound",
        ),
        (
            "Show organic cotton sheets queen size in white",
            "A",
            "manual",
            "cacheable product catalog query",
        ),
        (
            "Find refundable hotel rooms in Tokyo with breakfast included",
            "A",
            "manual",
            "cacheable travel inventory search under caller TTL policy",
        ),
        (
            "Show nonstop flights from SFO to Tokyo in November with premium economy",
            "A",
            "manual",
            "cacheable travel search under caller TTL policy",
        ),
        (
            "Find waterproof hiking boots size 10 in brown leather",
            "A",
            "manual",
            "cacheable retail product filter query",
        ),
        (
            "Show 4TB external SSDs under 300 USD with USB-C",
            "A",
            "manual",
            "cacheable product search with capacity and price constraints",
        ),
    ]
    examples = [
        RoutingExample(query=q, label=RoutingLabel.SEMANTIC_OK, slice_id=s, source=src, notes=notes)
        for q, s, src, notes in rows
    ]
    examples.extend(
        [
            RoutingExample(
                query="I'm having sudden panic attacks and chest pain after starting a new medication. What should I do?",
                label=RoutingLabel.SEMANTIC_OK,
                slice_id="G",
                source="legacy_eval",
                notes="reusable guidance prompt under a ttl-friendly namespace policy",
                namespace_policy="ttl_ok",
            ),
            RoutingExample(
                query="I think I swallowed a button battery and it feels stuck. What should I do?",
                label=RoutingLabel.SEMANTIC_OK,
                slice_id="G",
                source="legacy_eval",
                notes="reusable guidance prompt under a ttl-friendly namespace policy",
                namespace_policy="ttl_ok",
            ),
            RoutingExample(
                query="My throat is sore after travel. Could this often be due to a viral infection?",
                label=RoutingLabel.SEMANTIC_OK,
                slice_id="G",
                source="manual",
                notes="first-person reusable guidance prompt, not a personal record or live-state lookup",
                namespace_policy="ttl_ok",
            ),
            RoutingExample(
                query="If your throat gets sore after travel, is it often due to a viral infection?",
                label=RoutingLabel.SEMANTIC_OK,
                slice_id="G",
                source="manual",
                notes="same reusable guidance intent without personal scope",
                namespace_policy="ttl_ok",
            ),
            RoutingExample(
                query="I was traveling this week and now my throat is sore. Is that commonly caused by a virus?",
                label=RoutingLabel.SEMANTIC_OK,
                slice_id="G",
                source="manual",
                notes="TTL-friendly advice wording with a recent context mention",
                namespace_policy="ttl_ok",
            ),
            RoutingExample(
                query="After travel, what are common reasons someone might get a sore throat?",
                label=RoutingLabel.SEMANTIC_OK,
                slice_id="G",
                source="manual",
                notes="general advice paraphrase for travel-related symptom question",
                namespace_policy="ttl_ok",
            ),
        ]
    )
    examples.extend(_standalone_semantic_variation_examples())
    return examples


def _standalone_semantic_variation_examples() -> list[RoutingExample]:
    """Generated-style standalone query families kept separate from adversarial holdout rows.

    These examples teach the router that terse noun phrases, keyword queries, and adversarial
    standalone questions should still reach ANN. Reuse safety is handled by post-ANN gates.
    """

    rows = [
        ("capital city of Italy", "R", "generated", "terse factual lookup"),
        ("Germany capital city", "R", "generated", "keyword factual lookup"),
        ("which city is Italy's capital", "R", "generated", "reordered factual lookup"),
        ("population Japan", "R", "generated", "terse factual lookup"),
        ("List Meryl Streep filmography", "R", "generated", "standalone entertainment lookup"),
        ("Tom Cruise movies", "R", "generated", "terse standalone entity query"),
        ("TV shows with Bryan Cranston", "R", "generated", "standalone media-type query"),
        ("Python string reverse code", "R", "generated", "keyword code task"),
        ("reverse array javascript", "R", "generated", "keyword code task"),
        ("How do I sort a list in Python?", "R", "generated", "standalone code task"),
        ("JavaScript code to reverse an array", "R", "generated", "standalone code task"),
        ("How will I deploy a Node.js app to AWS?", "R", "generated", "future-tense instructional deployment query"),
        ("Steps to deploy Node.js to AWS", "R", "generated", "terse deployment how-to"),
        ("overview of World War I", "R", "generated", "history overview"),
        ("summarize the Cold War", "R", "generated", "history overview"),
        ("causes of the French Revolution", "R", "generated", "narrow standalone history query"),
        ("define dynamic programming", "R", "generated", "definition query"),
        ("Explain memoization to me", "R", "generated", "definition paraphrase"),
        ("example of recursion in code", "R", "generated", "example request"),
        ("when should I use iteration", "R", "generated", "use-case request"),
        ("basic bread recipe", "R", "generated", "terse recipe query"),
        ("how do I bake bread", "R", "generated", "recipe paraphrase"),
        ("how do I make a wooden table", "R", "generated", "standalone making task"),
        ("common flu symptoms", "R", "generated", "terse health information"),
        ("what are signs of dehydration", "R", "generated", "symptom information"),
        ("my throat is sore after travelling, maybe a virus", "R", "generated", "first-person reusable symptom guidance"),
        ("My throat is sore. I was travelling this week. Maybe a viral infection?", "R", "generated", "first-person reusable symptom guidance"),
        ("I was travelling recently and have a sore throat, could it be viral", "R", "generated", "first-person reusable symptom guidance"),
        ("after a trip I have a sore throat, maybe viral infection", "R", "generated", "first-person reusable symptom guidance"),
        ("how do I treat seasonal allergies", "R", "generated", "treatment information"),
        ("who runs Microsoft", "R", "generated", "current role paraphrase"),
        ("current GitHub CEO", "R", "generated", "terse current role query"),
        ("Who is the current CEO of GitHub?", "R", "generated", "current role question under caller freshness policy"),
        ("Current OpenAI CEO", "R", "generated", "terse current role query under caller freshness policy"),
        ("Who runs OpenAI?", "R", "generated", "current role paraphrase under caller freshness policy"),
        ("who founded Anthropic", "R", "generated", "founder query"),
        ("top cafes in Boston", "R", "generated", "terse local recommendation"),
        ("where should I eat in Chicago", "R", "generated", "recommendation paraphrase"),
        ("best bars in Seattle", "R", "generated", "venue recommendation"),
        ("Tokyo weather today", "R", "generated", "terse weather query under caller freshness policy"),
        ("temperature in Tokyo right now", "R", "generated", "weather paraphrase under caller freshness policy"),
    ]
    return [
        RoutingExample(query=q, label=RoutingLabel.SEMANTIC_OK, slice_id=s, source=src, notes=notes)
        for q, s, src, notes in rows
    ]


def _challenge_dispute_examples() -> list[RoutingExample]:
    """User pushback, correction, or confusion turns — always SKIP_CACHE."""

    rows = [
        # Short exclamations / confusion
        ("what?", "B", "manual", "short challenge turn"),
        ("huh?", "B", "manual", "short confusion turn"),
        ("wait", "B", "manual", "short challenge pause"),
        ("seriously?", "B", "manual", "verification pushback"),
        ("really?", "B", "manual", "verification pushback"),
        ("come on", "B", "manual", "frustration pushback"),
        ("ugh", "B", "manual", "frustration turn"),
        ("nope", "B", "manual", "rejection turn"),
        ("duh", "B", "manual", "sarcastic pushback"),
        ("Duh, try again", "B", "manual", "sarcastic retry request"),
        # Wrong / incorrect
        ("that's wrong", "B", "legacy_eval", "legacy correction/dispute style turn"),
        ("that is wrong", "B", "manual", "correction without contraction"),
        ("this is wrong", "B", "manual", "correction deictic"),
        ("it's wrong", "B", "manual", "correction contraction"),
        ("you're wrong", "B", "manual", "direct correction"),
        ("that's incorrect", "B", "manual", "formal correction"),
        ("incorrect", "B", "manual", "single-word correction"),
        ("what? that is wrong", "B", "manual", "challenge plus correction"),
        ("what? that's wrong", "B", "manual", "challenge plus correction variant"),
        ("no, that's not right", "B", "manual", "negated correction"),
        ("no that's wrong", "B", "manual", "negated correction terse"),
        ("that can't be right", "B", "manual", "doubt correction"),
        ("you got that wrong", "B", "manual", "agent-directed correction"),
        ("wrong answer", "B", "manual", "answer rejection"),
        # Retry / recheck
        ("recheck that", "B", "legacy_eval", "legacy correction/dispute style turn"),
        ("are you sure", "B", "legacy_eval", "legacy correction/dispute style turn"),
        ("are you sure?", "B", "manual", "verification question"),
        ("try again", "B", "manual", "retry request"),
        ("retry that", "B", "manual", "retry request variant"),
        ("redo that", "B", "manual", "retry request variant"),
        ("give it another try", "B", "manual", "soft retry request"),
        ("please try again", "B", "manual", "polite retry request"),
        # Nonsense / doesn't fit
        ("makes no sense", "B", "manual", "confusion rejection"),
        ("that makes no sense", "B", "manual", "confusion rejection"),
        ("this makes no sense", "B", "manual", "confusion rejection"),
        ("doesn't make sense", "B", "manual", "confusion rejection"),
        ("that doesn't make sense at all", "B", "manual", "strong confusion rejection"),
        ("not what I asked", "B", "manual", "intent mismatch pushback"),
        ("that's not what I asked for", "B", "manual", "intent mismatch pushback"),
        ("missed the point", "B", "manual", "quality pushback"),
        ("you missed the point", "B", "manual", "quality pushback"),
        # Softer doubt still bypasses cache
        ("I don't think that's right", "B", "manual", "soft correction"),
        ("that doesn't seem right", "B", "manual", "soft correction"),
        ("something seems off", "B", "manual", "soft doubt"),
        ("that answer looks wrong", "B", "manual", "answer quality dispute"),
        ("can you double check that", "B", "manual", "verification without accepting prior answer"),
        ("double check that", "B", "manual", "verification terse"),
    ]
    return [
        RoutingExample(
            query=q,
            label=RoutingLabel.SKIP_CACHE,
            slice_id=s,
            source=src,
            notes=notes,
            thread_scope_present=True,
        )
        for q, s, src, notes in rows
    ]


def _skip_cache_examples() -> list[RoutingExample]:
    rows = [
        ("Delete ticket 12345", "B", "manual", "destructive action"),
        ("Cancel my subscription right now", "B", "manual", "mutating and personalized"),
        ("Update customer 55291 email to new@example.com", "B", "manual", "account mutation"),
        ("Switch the deployment to the previous version", "B", "manual", "operational mutation"),
        ("Show today's revenue in apac", "E", "legacy_eval", "freshness sensitive"),
        ("What is the latest incident status?", "E", "manual", "freshness sensitive"),
        ("Give me the current CPU usage for host db-prod-1", "E", "manual", "live system status"),
        ("What changed this week in the audit log?", "E", "manual", "recent temporal request"),
        ("Show my open support cases", "B", "manual", "personalized account scoped"),
        ("Restart the failed job for me", "B", "manual", "action execution request"),
        ("Fix that", "B", "manual", "underspecified action request"),
        ("What is happening now?", "E", "manual", "underspecified and freshness sensitive"),
        ("Remove that user from the project", "B", "manual", "mutating action"),
        ("Show the latest revenue for my team", "E", "manual", "freshness plus personalization"),
        ("List top 5 revenue in apac today", "E", "legacy_eval", "legacy freshness-sensitive analytics query"),
        ("List top 5 revenue in apac now", "E", "legacy_eval", "legacy freshness variant"),
        ("List top 5 widget sales in us today", "E", "legacy_eval", "legacy realtime analytics query"),
        ("List top 5 macbook revenue in us today", "E", "legacy_eval", "legacy realtime analytics query"),
        (
            "I've had the same headache and congestion, but now I'm also having blurry vision. Could this still be sinusitis?",
            "G",
            "legacy_eval",
            "near-neighbor trap: added red-flag symptom changes reuse safety",
        ),
        (
            "The package already shipped. Can I reroute it to another city instead?",
            "G",
            "legacy_eval",
            "near-neighbor trap: similar policy topic but materially different action/state",
        ),
        (
            "Can I change the items in that order instead of cancelling it?",
            "G",
            "legacy_eval",
            "near-neighbor trap: mutation relative to a reusable policy question",
        ),
        (
            "Reset the production database to yesterday's snapshot",
            "B",
            "manual",
            "destructive operational action",
        ),
        (
            "Rotate the api keys for my workspace now",
            "B",
            "manual",
            "mutating credential action",
        ),
        (
            "Show me the current balance for my checking account",
            "E",
            "manual",
            "personalized and freshness-sensitive",
        ),
        (
            "What incidents are active right now for customer C777?",
            "E",
            "manual",
            "freshness plus customer-specific anchor",
        ),
        (
            "Delete the deployment and recreate it with the same settings",
            "B",
            "manual",
            "destructive change request",
        ),
        (
            "Cancel my order A123 right now",
            "B",
            "manual",
            "action request with anchor and urgency",
        ),
        (
            "Change the shipping address on my order to Tokyo",
            "B",
            "manual",
            "mutating request with personal scope",
        ),
        (
            "Restart my failed sync job now",
            "B",
            "manual",
            "operational action with personal scope and urgency",
        ),
        (
            "Remove jane.doe@example.com from my workspace",
            "B",
            "manual",
            "account-scoped execution request with identifier",
        ),
        (
            "Show me my active incidents right now",
            "E",
            "manual",
            "freshness plus personalized scope",
        ),
        (
            "What is my current order status for A123?",
            "E",
            "manual",
            "personalized and freshness-sensitive anchored query",
        ),
        (
            "Give me today's balance for account 998877",
            "E",
            "manual",
            "freshness-sensitive anchored query",
        ),
        (
            "Show transactions posted this week for account 998877",
            "E",
            "manual",
            "account-specific recent activity",
        ),
        (
            "What wire transfers were sent today from my business account?",
            "E",
            "manual",
            "personalized fresh banking activity",
        ),
        (
            "Show the live exchange rate used for USD to JPY right now",
            "E",
            "manual",
            "live market data",
        ),
        (
            "What errors appeared in the production logs this month?",
            "E",
            "manual",
            "mutable operational logs with time window",
        ),
        (
            "Show failed login attempts for my account this week",
            "E",
            "manual",
            "personalized security log activity",
        ),
        (
            "Apply the same fix to production now",
            "B",
            "manual",
            "underspecified action execution",
        ),
        (
            "Summarize the HIPAA guidelines for patient data sharing.",
            "H",
            "queries_txt",
            "PHI / regulatory narrative — avoid broad semantic reuse",
        ),
        (
            "Give me a rundown of the regulatory requirements for PHI disclosure under HIPAA.",
            "H",
            "queries_txt",
            "PHI paraphrase — align with skip-cache sibling",
        ),
        (
            "What is the current wait time at the St. Jude's ER right now?",
            "H",
            "queries_txt",
            "explicit freshness — should not route SEMANTIC_OK",
        ),
        (
            "How long is the wait at the emergency room at St. Jude's?",
            "H",
            "queries_txt",
            "freshness paraphrase — align label with prior query",
        ),
        (
            "Which patients are ineligible for the clinical trial?",
            "H",
            "queries_txt",
            "polarity flip trap vs eligibility query",
        ),
        (
            "Check my latest blood glucose reading from my Dexcom.",
            "H",
            "queries_txt",
            "personal device / individualized",
        ),
        (
            "What was my most recent sugar level?",
            "H",
            "queries_txt",
            "personalized health metric",
        ),
        (
            "What is the recommended dosage for Pediatric Amoxicillin?",
            "H",
            "queries_txt",
            "pediatric dosing — unsafe to reuse with adult dosing paraphrase",
        ),
        (
            "What is the recommended dosage for Adult Amoxicillin?",
            "H",
            "queries_txt",
            "adult dosing sibling — label with pediatric pair as SKIP_CACHE",
        ),
        (
            "How do I treat a Type 1 diabetic with low blood sugar?",
            "H",
            "queries_txt",
            "diabetes subtype-specific — unsafe semantic reuse vs Type 2 sibling",
        ),
        (
            "How do I treat a Type 2 diabetic with low blood sugar?",
            "H",
            "queries_txt",
            "Type 2 sibling — label with Type 1 pair as SKIP_CACHE",
        ),
    ]
    examples = [
        RoutingExample(query=q, label=RoutingLabel.SKIP_CACHE, slice_id=s, source=src, notes=notes)
        for q, s, src, notes in rows
    ]
    for idx, example in enumerate(examples):
        if example.query in {
            "Show today's revenue in apac",
            "What is the latest incident status?",
            "Give me the current CPU usage for host db-prod-1",
            "What changed this week in the audit log?",
            "Show the latest revenue for my team",
            "List top 5 revenue in apac today",
            "List top 5 revenue in apac now",
            "List top 5 widget sales in us today",
            "List top 5 macbook revenue in us today",
            "Show me my active incidents right now",
            "What is my current order status for A123?",
            "Give me today's balance for account 998877",
            "Show transactions posted this week for account 998877",
            "What wire transfers were sent today from my business account?",
            "Show the live exchange rate used for USD to JPY right now",
            "What errors appeared in the production logs this month?",
            "Show failed login attempts for my account this week",
        }:
            examples[idx] = RoutingExample(
                query=example.query,
                label=example.label,
                slice_id=example.slice_id,
                source=example.source,
                notes=example.notes,
                thread_scope_present=example.thread_scope_present,
                namespace_policy="freshness_strict",
            )
    examples.extend(_challenge_dispute_examples())
    return examples


def _exact_only_examples() -> list[RoutingExample]:
    rows = [
        ("Lookup order #A123 status", "C", "legacy_eval", "anchored order id"),
        ("Find order A123 status", "C", "legacy_eval", "anchored order id paraphrase"),
        ("Search ticket INC-4432", "C", "manual", "anchored incident id"),
        ("Show customer C777 profile", "C", "legacy_eval", "anchored customer id"),
        ("What happened on host db-prod-1.example.com?", "C", "manual", "hostname anchor"),
        ("Lookup incident 884221", "C", "manual", "long numeric identifier"),
        ("Find account 998877 current plan", "C", "manual", "anchored account query"),
        ("What is the status of ticket 55192?", "C", "manual", "ticket id query"),
        ("Search order Z999 status", "C", "legacy_eval", "identifier paraphrase"),
        ("Find uuid 123e4567-e89b-12d3-a456-426614174000", "C", "manual", "uuid anchored"),
        ("Show host prod-api-7.company.net details", "C", "manual", "hostname anchor"),
        ("Open case 732001", "C", "manual", "case identifier"),
        ("Find customer C901 profile", "C", "legacy_eval", "customer anchor"),
        ("Lookup ticket 145337 assigned owner", "C", "manual", "ticket anchor"),
        ("Lookup customer #C777 profile", "C", "legacy_eval", "legacy anchored customer query"),
        ("find customer c777 profile", "C", "legacy_eval", "legacy anchored customer paraphrase"),
        ("search customer C777 profile", "C", "legacy_eval", "legacy anchored customer paraphrase"),
        ("Lookup order #Z999 status", "C", "legacy_eval", "legacy anchored order query"),
        ("find order z999 status", "C", "legacy_eval", "legacy anchored order paraphrase"),
        ("search order Z999 status", "C", "legacy_eval", "legacy anchored order paraphrase"),
        ("Lookup customer #C901 profile", "C", "legacy_eval", "legacy anchored customer query"),
        ("search for customer C901 profile", "C", "legacy_eval", "legacy anchored customer paraphrase"),
        ("Check invoice INV-2024-9912", "C", "manual", "invoice identifier"),
        ("Show me order B456 tracking status", "C", "manual", "order identifier"),
        ("Look up ticket INC-99881 owner", "C", "manual", "incident id"),
        ("Find host db-stage-12.internal.example.net", "C", "manual", "hostname identifier"),
        ("Search user jane.doe@example.com account state", "C", "manual", "email identifier"),
        ("Get contract 440981 renewal status", "C", "manual", "long numeric anchor"),
        ("Show me the last lab results for patient ID: 88291.", "C", "queries_txt", "patient identifier — exact/anchor style"),
        ("What were the laboratory findings for patient ID: 11022?", "C", "queries_txt", "patient identifier paraphrase"),
    ]
    return [RoutingExample(query=q, label=RoutingLabel.EXACT_ONLY, slice_id=s, source=src, notes=notes) for q, s, src, notes in rows]


def _thread_scoped_examples() -> list[RoutingExample]:
    rows = [
        ("What about that one?", "D", "manual", "depends on prior referent"),
        ("Same but in december", "D", "manual", "underspecified mutation"),
        ("Instead use tokyo", "D", "manual", "follow-up mutation"),
        ("Do it again", "D", "manual", "thread dependent repeat"),
        ("Now do same for europe", "D", "legacy_eval", "follow-up mutation from legacy data"),
        ("Also for apac", "D", "legacy_eval", "follow-up mutation from legacy data"),
        ("Use premium economy instead", "D", "manual", "follow-up constraint change"),
        ("What about the second option?", "D", "manual", "requires thread history"),
        ("Try the same query for last month", "D", "manual", "relative follow-up mutation"),
        ("Re-run that with tokyo", "D", "manual", "thread-scoped mutation"),
        ("Do the same for us", "D", "legacy_eval", "legacy mutation"),
        ("What about that customer?", "D", "manual", "ambiguous referent"),
        ("Make it the same as before but weekly", "D", "manual", "underspecified stateful follow-up"),
        ("Instead use the earlier dataset", "D", "manual", "follow-up depends on prior state"),
        ("Now do same for us", "D", "legacy_eval", "legacy follow-up mutation"),
        ("Now do same for emea", "D", "legacy_eval", "legacy follow-up mutation"),
        ("Also for europe", "D", "legacy_eval", "legacy follow-up mutation"),
        ("go back to europe", "D", "legacy_eval", "legacy branch / undo region pivot"),
        ("the first one", "D", "legacy_eval", "legacy underspecified ordinal referent"),
        ("No, I want to avoid carrying a balance entirely. Does that change your recommendation?", "D", "legacy_eval", "legacy follow-up clarification"),
        ("Also wanted to know if I should keep all my money in one place?", "D", "legacy_eval", "legacy under-specified follow-up"),
        ("No, I mean only for day-to-day spending money.", "D", "legacy_eval", "legacy follow-up clarification"),
        ("No, these are happening every day now.", "D", "legacy_eval", "legacy follow-up clarification"),
        ("That doesn't seem to match what I'm feeling.", "D", "legacy_eval", "legacy disagreement follow-up"),
        ("That doesn't sound right.", "D", "legacy_eval", "legacy disagreement follow-up"),
        ("Use the second region instead", "D", "manual", "requires previous options"),
        ("Keep everything the same but switch to monthly", "D", "manual", "follow-up mutation"),
        ("What about the earlier incident?", "D", "manual", "ambiguous prior referent"),
        ("Do that again but for the first customer", "D", "manual", "stateful follow-up"),
        ("Same question, but for the other plan", "D", "manual", "requires earlier referent"),
        (
            "Can you retrieve the discharge summary for John Doe?",
            "D",
            "queries_txt",
            "patient-specific record — thread / scope",
        ),
        (
            "I need the discharge summary for Jane Smith.",
            "D",
            "queries_txt",
            "patient-specific record paraphrase",
        ),
    ]
    return [
        RoutingExample(
            query=q,
            label=RoutingLabel.THREAD_SCOPED_ONLY,
            slice_id=s,
            source=src,
            notes=notes,
            thread_scope_present=True,
        )
        for q, s, src, notes in rows
    ]
