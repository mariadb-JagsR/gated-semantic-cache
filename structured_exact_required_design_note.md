# Structured-Exact-Required Cache Path Design Note

## Purpose

This document defines a **specific cache path** for requests that have already been classified as:

- `STRUCTURED_EXACT_REQUIRED`

This path is for requests where:

- semantic retrieval may still be useful for candidate generation,
- but **semantic similarity must never be treated as proof of safe reuse**, and
- final reuse requires equality on extracted **answer-critical structured constraints**.

This note is intentionally narrow. It does **not** describe the full semantic cache system. It only defines the behavior after the request has already been routed into this path.

---

## Why this path exists

Some requests are semantically very close in vector space but differ in a way that changes the correct answer.

Examples:

- `show me all pants size 32x32, cotton, stretch, under $40, brown`
- `show me all pants size 32x34, cotton, stretch, under $40, brown`

These two queries will likely be close neighbors under a good embedding model because they are almost the same request.

However, reusing the cached answer from one for the other would be wrong because `32x32` and `32x34` are answer-critical differences.

The same issue appears in many domains:

- flights: `1 checked bag` vs `2 checked bags`
- hotels: `2 guests` vs `4 guests`
- finance: `under $5,000` vs `under $50,000`
- SQL/data: `last 7 days` vs `last 30 days`
- DB config/help: `READ COMMITTED` vs `REPEATABLE READ`

The design rule for this path is:

> semantic retrieval may retrieve candidates, but only structured critical-constraint equality can authorize reuse.

---

## Scope

This path should be used only after a previous routing/classification step has already determined that:

1. the query is not a plain `SEMANTIC_OK` request,
2. the query is not `SKIP_CACHE`, and
3. the query requires **exactness over extracted constraints**.

Typical triggers for this class:

- filtered search requests
- requests with multiple attributes
- price/range bounded requests
- size/dimension constrained requests
- time-window constrained requests
- count/quantity constrained requests
- queries whose answer depends on a small change in one constraint

This path should not try to solve arbitrary conversation reconstruction.

---

## Core design principle

### Allowed

- semantic ANN retrieval for candidate generation
- generic normalization
- generic typed constraint extraction
- canonical structured key generation when confidence is high
- exact comparison of answer-critical constraints before reuse

### Forbidden

- using vector similarity alone to authorize reuse
- using “very high similarity” as final proof of equivalence
- assuming better embeddings or more dimensions solve correctness
- using a hot-path LLM judge for normal operation
- reconstructing large semantic state objects before every lookup

---

## High-level request flow

```text
incoming query
  -> request already classified as STRUCTURED_EXACT_REQUIRED
  -> normalize query
  -> generic typed constraint extraction
  -> if structured extraction confidence is high:
         build canonical structured key
         exact structured lookup
         if hit: return
     else:
         skip structured exact lookup
  -> semantic retrieval for candidate generation (optional but allowed)
  -> extract/compare critical constraints against candidates
  -> if exact critical-constraint match: return cached response
  -> else: miss
  -> live answer path
  -> optional insert if policy allows
```

Important:

- structured exact lookup is preferred when extraction confidence is high,
- semantic retrieval is secondary and may help discover neighbors,
- but final reuse must pass exact constraint matching.

---

## Definitions

### Structured-exact-required

A request class where one or more structured constraints are answer-critical.

### Answer-critical constraint

A normalized constraint whose change can produce a meaningfully different answer set or response.

Examples:

- size
- quantity
- numeric upper/lower bound
- date range
- count of passengers/guests/items
- product dimension
- time window
- binary option such as refundable / nonstop / in-stock / enabled

### Candidate generation

Using embeddings and ANN retrieval to find nearby prior cache entries.

### Reuse authorization

The final decision to serve a cached answer. In this path, reuse authorization must depend on critical-constraint equality, not vector similarity alone.

---

## Design overview

This path has three core stages:

1. **Generic normalization**
2. **Typed constraint extraction**
3. **Constraint-authorized reuse**

### Stage 1: Generic normalization

Normalize the surface form of the query so that common equivalent expressions map to the same representation where possible.

Required normalization steps:

- lowercase
- trim and collapse whitespace
- normalize punctuation where safe
- normalize Unicode
- normalize currency and numeric formatting where possible
- standardize common range phrases
- standardize dimension separators
- standardize simple plural/singular variants where safe

Examples:

- `under $40` -> `price <= 40 usd`
- `less than 40 dollars` -> `price <= 40 usd`
- `32 X 32` -> `32x32`
- `brown-colored` -> `brown`
- `last seven days` -> `last 7 days`

Normalization should remain generic and cheap.

### Stage 2: Typed constraint extraction

Extract a small, generic intermediate representation of constraints from the normalized query.

Do **not** attempt full semantic parsing.

Do **not** require deep domain ontology.

The extractor should identify common typed constraints that recur across many domains.

Required generic constraint families:

- anchor terms
- identifiers
- numeric bounds
- ranges
- dimensions
- quantities/counts
- dates/time windows
- categorical values
- binary attributes / flags
- sort indicators when relevant

The goal is not to fully understand the user’s domain. The goal is to capture enough answer-critical structure to prevent unsafe reuse.

### Stage 3: Constraint-authorized reuse

A candidate may only be reused if:

- namespace and metadata filters are compatible,
- candidate is not expired,
- extraction from the current request produced sufficient critical constraints,
- the same critical constraints can be compared against the candidate’s stored normalized constraints,
- all required critical constraints are equal under normalization.

If critical equality cannot be established, the candidate must not be reused.

---

## Intermediate representation

Use a generic typed intermediate representation rather than domain-specific semantic objects.

Suggested structure:

```json
{
  "shape": "filtered_search",
  "anchors": ["pants"],
  "constraints": [
    {"kind": "dimension", "name": "size", "value": "32x32", "confidence": 0.98, "critical": true},
    {"kind": "categorical", "name": "material", "value": "cotton", "confidence": 0.99, "critical": true},
    {"kind": "categorical", "name": "attribute", "value": "stretch", "confidence": 0.94, "critical": true},
    {"kind": "numeric_bound", "name": "price", "op": "<=", "value": 40, "unit": "usd", "confidence": 0.99, "critical": true},
    {"kind": "categorical", "name": "color", "value": "brown", "confidence": 0.99, "critical": true}
  ],
  "confidence": 0.96
}
```

Notes:

- `shape` is optional but useful.
- `critical` marks constraints that must match before reuse.
- `confidence` is local to each constraint and also available at the whole-object level.
- this representation must stay generic and compact.

---

## Required extractor modules

Build the extractor as a set of small, generic modules.

### 1. Numeric/range extractor

Recognize patterns such as:

- under $40
- less than 100
- greater than 10
- at least 2
- between 5 and 8
- no more than 30 days

Normalize to typed bounds.

### 2. Dimension extractor

Recognize patterns such as:

- 32x32
- 10x12
- 15 inch
- 64 GB
- size 11
- XL

Normalize where possible into a stable value form.

### 3. Date/time extractor

Recognize patterns such as:

- today
- current month
- last 7 days
- in november
- next week
- before June 1

Normalize to typed temporal constraints.

### 4. Identifier extractor

Recognize patterns such as:

- UUID-like values
- order numbers
- ticket IDs
- emails
- hostnames
- long stable alphanumeric strings

If strong identifiers are present, they may dominate constraint matching.

### 5. Categorical matcher

Recognize common category-like and attribute-like values using generic dictionaries and optional pluggable domain dictionaries.

Examples of generic dictionaries:

- colors
- common materials
- boolean attributes like stretch / refundable / nonstop / enabled
- common sort words like cheapest / newest / highest rated

### 6. Quantity/count extractor

Recognize values like:

- 2 guests
- 4 passengers
- 1 bag
- 10 rows
- top 5

Counts are often answer-critical and must not be dropped.

---

## Confidence model

This path must not assume extraction is always complete.

### High-confidence extraction

Extraction is considered high-confidence when:

- at least one anchor or trusted namespace exists,
- at least one critical constraint is extracted cleanly,
- the extracted constraints are internally consistent,
- no strong ambiguity markers dominate the query,
- normalization succeeded for the critical tokens.

### Low-confidence extraction

If extraction is incomplete or low-confidence:

- do not build a structured exact key,
- semantic retrieval may still be used for candidate generation,
- but reuse must only happen if critical equality can still be established,
- otherwise the request must miss.

Important:

Low-confidence extraction should degrade to **fewer cache hits**, not riskier ones.

---

## Canonical structured key

### Purpose

The canonical structured key provides a deterministic lookup path for high-confidence structured requests.

### Key rules

- only include high-confidence normalized constraints,
- include anchor/category only when reliable,
- sort constraints deterministically,
- include namespace/context when relevant,
- do not include low-confidence inferred semantics.

### Example

Input:

`show me all pants size 32x32, cotton, stretch, under $40, brown`

Normalized structured key:

```text
filtered_search|pants|attribute:stretch|color:brown|material:cotton|price<=40usd|size:32x32
```

A variant with `32x34` must produce:

```text
filtered_search|pants|attribute:stretch|color:brown|material:cotton|price<=40usd|size:32x34
```

This guarantees that structured exact lookup will miss cleanly when answer-critical constraints differ.

### When not to build the key

Do not build a structured key if:

- no reliable anchor/category exists and namespace is insufficient,
- all extracted constraints are low-confidence,
- the query is heavily ambiguous,
- constraint forms are too incomplete to normalize.

---

## Serving path behavior

### Step 1: Normalize

Run generic normalization.

### Step 2: Extract constraints

Produce a typed constraint representation.

### Step 3: Attempt structured exact lookup if confidence is high

If the extracted representation is high-confidence:

- build the canonical structured key,
- attempt exact structured cache lookup,
- if hit: return immediately.

### Step 4: Semantic retrieval for candidates

If exact structured lookup misses, semantic retrieval is allowed for candidate generation.

Use embeddings + FAISS HNSW to retrieve nearby cached entries.

### Step 5: Candidate compatibility filtering

Filter candidates by:

- namespace
- version metadata
- expiration
- thread scope if required
- optional source/corpus version

### Step 6: Critical-constraint comparison

For each candidate, compare the current request’s critical constraints to the candidate’s stored normalized constraints.

A candidate is reusable only if:

- all required critical constraints match exactly after normalization,
- no critical constraint is missing on one side when required,
- no critical value differs.

### Step 7: Reuse or miss

- if a candidate passes: return cached response,
- else: miss and go live.

---

## Semantic retrieval rules in this path

Semantic retrieval is useful here, but only as a helper.

### Allowed uses

- finding nearby prior queries
- surfacing possibly reusable candidates
- improving recall where exact phrasing differs

### Forbidden uses

- final proof of equivalence
- “high similarity bypass”
- threshold-only authorization
- higher-dimensional embeddings as a substitute for exact constraint checking

### Important clarification

Better embeddings may improve candidate recall.
They do **not** remove the need for constraint equality.

---

## Candidate constraint comparison

Constraint comparison should operate on the normalized typed representation.

### Matching rules

For each `critical=true` constraint in the current request:

- candidate must contain the same normalized constraint name,
- candidate must contain the same normalized operator if relevant,
- candidate must contain the same normalized value,
- unit must match if relevant,
- dimension token must match exactly after normalization,
- date/time windows must match in normalized form,
- counts and quantities must match exactly.

### Examples

#### Reuse allowed

Current:
- size = 32x32
- price <= 40 usd
- color = brown

Candidate:
- size = 32x32
- price <= 40 usd
- color = brown

Allowed if all other required metadata matches.

#### Reuse forbidden

Current:
- size = 32x34

Candidate:
- size = 32x32

Forbidden even if similarity is extremely high.

#### Reuse forbidden

Current:
- bags = 2

Candidate:
- bags = 1

Forbidden.

#### Reuse forbidden

Current:
- last 30 days

Candidate:
- last 7 days

Forbidden.

---

## Storage requirements for this path

Each cache entry in this path must store:

- original query text
- normalized query text
- embedding vector
- namespace
- response payload
- timestamps / expiration
- canonical structured key if available
- normalized typed constraint representation
- set of critical constraints
- optional anchor/category
- version metadata
- source/corpus metadata as relevant

The stored constraint representation must be compact and stable.

Do not store large semantic analysis objects.

---

## Insert policy

A live response may be inserted into this path only if:

- classification was `STRUCTURED_EXACT_REQUIRED`,
- extraction produced a valid typed constraint representation,
- at least one critical constraint was identified,
- namespace and metadata are available,
- response is cacheable by broader system policy,
- the entry is not freshness-unsafe or session-private unless explicitly scoped.

### Insert preference

If structured extraction confidence is high:

- store the canonical structured key,
- store normalized constraints,
- store the embedding.

If extraction confidence is low:

- either do not insert,
- or insert only if system policy explicitly allows low-confidence semantic-only candidate storage.

For v1, prefer conservative insert behavior for low-confidence structured extraction.

---

## Failure behavior

This path should fail conservatively but simply.

### If normalization fails partially

Use whatever safe normalized tokens remain.

### If structured extraction fails

- do not build structured key,
- semantic retrieval may still occur,
- but no candidate may be reused unless critical equality can still be established.

### If constraint comparison cannot be completed

Treat as mismatch.

### Important rule

Failure must reduce reuse, not trigger fallback unsafe reuse.

---

## Non-goals for this path

This path must **not** attempt to:

- fully understand arbitrary user intent,
- reconstruct multi-turn hidden state,
- learn a full domain ontology,
- compare candidates using an LLM on the hot path,
- infer missing critical constraints from vague context,
- convert near-neighbor similarity into final authorization.

---

## Evaluation requirements

This path must be evaluated with focused structured-difference benchmarks.

### Required benchmark categories

1. same query, same constraints, paraphrased wording
2. same query with one critical dimension changed
3. same query with one numeric bound changed
4. same query with one quantity changed
5. same query with one date window changed
6. same query with one binary option changed
7. incomplete structured extraction cases
8. low-confidence extraction fallback cases

### Required metrics

- structured exact-key hit rate
- semantic candidate retrieval recall
- reuse precision after constraint matching
- wrong reuse rate
- miss rate due to extraction incompleteness
- latency of normalization/extraction/ANN/comparison

The most important metric is:

> wrong reuse rate must remain near zero for critical-constraint differences.

---

## Acceptance criteria

The implementation is acceptable only if it satisfies all of the following:

1. changing a critical constraint changes the canonical structured key when high-confidence extraction exists,
2. semantic retrieval may still return nearby candidates,
3. nearby candidates are **not** reused when any critical constraint differs,
4. no hot-path LLM verification is required,
5. no threshold-only bypass exists,
6. failure to extract constraints causes a miss rather than unsafe reuse,
7. the code and data model remain generic and do not hard-code one business domain.

---

## Pseudocode

### Serving path

```python
def handle_structured_exact_required(query, context):
    normalized = normalize_query(query)
    extracted = extract_typed_constraints(normalized, context)

    if extracted.confidence >= context.structured_key_threshold and extracted.has_critical_constraints():
        structured_key = build_structured_key(extracted, context)
        if structured_key is not None:
            exact_hit = structured_cache.get(structured_key)
            if exact_hit is not None:
                return exact_hit

    embedding = embed(normalized)
    candidates = faiss_search(embedding, namespace=context.namespace, top_k=5)

    filtered = filter_candidates(candidates, context)

    for candidate in filtered:
        if critical_constraints_match(extracted, candidate.extracted_constraints):
            return candidate.response

    live = live_answer(query, context)
    maybe_insert_structured_entry(query, normalized, extracted, live, context)
    return live
```

### Constraint comparison

```python
def critical_constraints_match(current, candidate):
    if current is None or candidate is None:
        return False

    current_critical = [c for c in current.constraints if c.critical]
    if not current_critical:
        return False

    candidate_index = index_constraints(candidate.constraints)

    for c in current_critical:
        other = candidate_index.get((c.kind, c.name))
        if other is None:
            return False
        if normalize_constraint_value(c) != normalize_constraint_value(other):
            return False
        if getattr(c, 'op', None) != getattr(other, 'op', None):
            return False
        if getattr(c, 'unit', None) != getattr(other, 'unit', None):
            return False

    return True
```

---

## Implementation notes

### Keep generic, not magical

The extractor library should aim for generic typed constraint detection, not universal semantic interpretation.

### Optional domain packs

The design may support optional domain dictionaries or domain packs later, but the core path must remain useful without them.

### Start simple

Implement the following first:

- normalization
- numeric/range extraction
- dimension extraction
- date/time extraction
- identifier extraction
- simple categorical dictionaries
- canonical structured key
- critical-constraint comparison

Do not start with:

- full parser frameworks
- LLM extraction
- large transformer sequence taggers
- ontology engines

---

## Final design stance

This path exists because some requests are semantically close but structurally different in answer-critical ways.

The correct design is:

- use semantics to find candidates,
- use structured critical equality to authorize reuse.

If the implementation drifts toward treating vector similarity as proof of correctness, the design has failed.
