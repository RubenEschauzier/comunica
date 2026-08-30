You are completely right about Option B. In decentralized/adaptive query processing (like Link Traversal or Solid), $TP_3$ and $TP_4$ are streaming data from remote sources. If the router opportunistically prioritizes $CR$, the intermediate tuples produced by $TP_3$ and $TP_4$ would be starved and never joined to completion, leading to **incomplete results**.

This confirms that **Option A (concurrent execution of exclusive plans)** is the exact model needed for your research:
- The **Base Plan** must continue running concurrently to process all tuples from $TP_3$ and $TP_4$.
- The **Composite Plan** runs concurrently to process tuples using $CR$.

---

### What happens when $CR_2$ is added (and can combine with $CR_1$)?

Let's trace this with a concrete example:
- Query: $TP_1 \bowtie TP_2 \bowtie TP_3 \bowtie TP_4$
- $CR_1$ covers $TP_3 \bowtie TP_4$
- $CR_2$ covers $TP_1 \bowtie TP_2$

Notice that $CR_1$ and $CR_2$ cover **disjoint** parts of the query. That means they can be combined into a direct join $CR_1 \bowtie CR_2$!

#### The 4 Concurrent Plans
Because $CR_1$ and $CR_2$ address independent sub-queries, they produce **$2 \times 2 = 4$ valid execution plans**:

| Plan ID | Exclusive Choice for $\{TP_3, TP_4\}$ | Exclusive Choice for $\{TP_1, TP_2\}$ | Active Operators in Plan |
| :--- | :--- | :--- | :--- |
| **Plan 0** | Base ($TP_3, TP_4$) | Base ($TP_1, TP_2$) | $\{TP_1, TP_2, TP_3, TP_4\}$ |
| **Plan 1** | Composite ($CR_1$) | Base ($TP_1, TP_2$) | $\{TP_1, TP_2, CR_1\}$ |
| **Plan 2** | Base ($TP_3, TP_4$) | Composite ($CR_2$) | $\{CR_2, TP_3, TP_4\}$ |
| **Plan 3** | Composite ($CR_1$) | Composite ($CR_2$) | $\{CR_1, CR_2\}$ |

---

### How Plans Combine Dynamically in `addOperator`

When a new composite resource arrives at runtime, it **spawns new plans by branching from compatible existing plans**:

1. **Initially (Only Base)**:
   - `Plan 0`: $\{TP_1, TP_2, TP_3, TP_4\}$

2. **When $CR_1$ ($TP_3 \bowtie TP_4$) arrives**:
   - It checks existing plans. Plan 0 does not use $CR_1$.
   - It clones Plan 0 and replaces $\{TP_3, TP_4\}$ with $CR_1$:
   - Creates **`Plan 1`**: $\{TP_1, TP_2, CR_1\}$.

3. **When $CR_2$ ($TP_1 \bowtie TP_2$) arrives**:
   - It checks existing plans:
     - Plan 0 has no overlap with $CR_2 \to$ clones Plan 0 into **`Plan 2`**: $\{CR_2, TP_3, TP_4\}$.
     - **Plan 1 has no overlap with $CR_2$!** $\to$ clones Plan 1 into **`Plan 3`**: $\{CR_1, CR_2\}$!
   - *What if an incompatible $CR_3$ ($TP_2 \bowtie TP_3$) arrived instead?*
     - It overlaps with both $CR_1$ ($TP_3$) and $CR_2$ ($TP_2$).
     - It could only clone Plan 0 (producing $\{TP_1, CR_3, TP_4\}$). It could **never** clone Plan 1, 2, or 3.

---

### How Tuples Route Inside Each Plan (With `planId`)

By stamping intermediate tuples with their `planId`, every plan is guaranteed plan correctness and adaptivity:

#### Inside Plan 3 (Combined: $\{CR_1, CR_2\}$)
- When $CR_1$ produces a tuple (`done = {TP_3, TP_4}, planId = 3`):
  - In Plan 3, the only remaining operator is $CR_2$.
  - It routes directly to $CR_2$.
  - Result: $CR_1 \bowtie CR_2$ completed in one join!

#### Inside Plan 1 (Composite $CR_1$: $\{TP_1, TP_2, CR_1\}$)
- At state $TP_1$, Plan 1's candidate list is: `[ TP_2, CR_1 ]`.
- Plan 1 can adaptively choose to **filter through $TP_2$ first**!
- $TP_1 \bowtie TP_2$ is computed, producing `done = {TP_1, TP_2}, planId = 1`.
- At state $\{TP_1, TP_2\}$, Plan 1's candidate list is: `[ CR_1 ]`.
- It routes to $CR_1$.
- **It never touches $TP_3$ or $TP_4$**, because those belong to Plan 0.

#### Inside Plan 0 (Pure Base: $\{TP_1, TP_2, TP_3, TP_4\}$)
- Processes all tuples from $TP_3$ and $TP_4$.
- Never touches $CR_1$ or $CR_2$.
- Guarantees that no remote base data is ignored.

---

### The Concrete Implementation Steps

1. **In [`IStemsBindingsMetadata`](file:///home/ruben-eschauzier/projects/querying-derived-resources/comunica-adaptive-derived-resources/packages/actor-rdf-join-inner-multi-stems/lib/StemsControllerStream.ts#L430)**:
   Add `planId?: number`.
2. **In [`StemsControllerStream.read()`](file:///home/ruben-eschauzier/projects/querying-derived-resources/comunica-adaptive-derived-resources/packages/actor-rdf-join-inner-multi-stems/lib/StemsControllerStream.ts#L195)**:
   - If an incoming tuple has `planId: p`, it **only** executes `nextRoutes[p]`.
   - If an incoming tuple is untagged (`planId === undefined`), it forks across all active plans, tagging each copy with that plan's index.
3. **In [`BaseRouter.ts`](file:///home/ruben-eschauzier/projects/querying-derived-resources/comunica-adaptive-derived-resources/packages/actor-rdf-join-inner-multi-stems/lib/routers/BaseRouter.ts#L130)**:
   `routing[planId]` holds the candidates for that specific plan. Adding a CR clones compatible existing routes and substitutes the CR's operations, allowing compatible CRs to combine naturally into Plan 3 ($CR_1 \bowtie CR_2$).





**Yes, absolutely.** You have just described one of the most powerful concepts in adaptive query execution: **Work-Sharing / Deferred Splitting** (similar to CACQ/TelegraphCQ).

Here is the exact analysis of why the naive way does redundant work, why your idea is much more performant, and how to implement it cleanly with your current setup.

---

### 1. Does the naive way do extra work?
**Yes, a lot.**
If Plan 0 and Plan 1 both route to $TP_2$, pushing two separate tuples means $TP_2$ executes:
- **2x hash calculations** on the exact same bindings.
- **2x `tripleMap.get()`** lookups.
- **2x RDF term equality comparisons** in `funJoin` for every match.
- **2x new `Bindings` allocations** and context metadata creations.

If 3 or 4 plans share an early operator, you do 3x or 4x the CPU and memory work for the exact same join.

---

### 2. Can we collapse routing decisions? Is it more performant?
**Yes, it is dramatically more performant.**

Instead of having the operator output the tuple twice immediately, an even better technique is **Deferred Splitting (Lazy Forking)**:
- Tag the tuple with a bitmask of all plans it currently satisfies: e.g. `planMask = (1 << 0) | (1 << 1) = 3` (satisfies both Plan 0 and Plan 1).
- Push **one single tuple** to $TP_2$.
- $TP_2$ does the join **once**.
- $TP_2$ produces **one single output tuple** that still carries `planMask = 3`.
- **You only fork when the plans actually DIVERGE!**

#### Why Deferred Splitting is so powerful:
1. **Zero duplicate joins**: If Plan 0 and Plan 1 share multiple consecutive operators (e.g. $TP_1 \to TP_2 \to TP_5$), the tuple stays as **one single tuple** across all of them.
2. **Early filtering savings**: If $TP_2$ filters out the tuple (0 join matches), **zero tuples are ever duplicated**.
3. **No changes required in [`StemsOperatorStream.ts`](file:///home/ruben-eschauzier/projects/querying-derived-resources/comunica-adaptive-derived-resources/packages/actor-rdf-join-inner-multi-stems/lib/StemsOperatorStream.ts#L201)**: Look at line 201:
   ```typescript
   const copy = { ...this.matchMetadata! };
   copy.done |= this.doneBitMask;
   copy.order = [ ...copy.order, this.operatorIndex ];
   ```
   `StemsOperatorStream` already performs a shallow copy of `matchMetadata`. It automatically preserves `planMask` without modifying a single line of the operator!

---

### 3. How to implement this in `StemsControllerStream.read()`

All the collapsing and forking logic lives entirely inside [`StemsControllerStream.read()`](file:///home/ruben-eschauzier/projects/querying-derived-resources/comunica-adaptive-derived-resources/packages/actor-rdf-join-inner-multi-stems/lib/StemsControllerStream.ts#L185-L204):

```typescript
const partialResultMetadata = item.getContextEntry(stemsContextKeys.eddiesMetadata)!;
const nextRoutes = this.routingTable[partialResultMetadata.done];

if (nextRoutes === undefined || nextRoutes.length === 0) {
  return item;
}

// 1. Determine which plans this tuple is currently eligible for:
// If untagged (e.g. from common source), it is eligible for all active routes: (1 << N) - 1
const currentPlanMask = partialResultMetadata.planMask ?? ((1 << nextRoutes.length) - 1);

// 2. Group plans by target operator to COLLAPSE common routing decisions:
const operatorBatches = new Map<number, { joinVars: RDF.Variable[]; targetPlanMask: number }>();

for (let planIdx = 0; planIdx < nextRoutes.length; planIdx++) {
  // Check if this tuple belongs to this plan:
  if ((currentPlanMask & (1 << planIdx)) === 0) {
    continue;
  }

  const route = nextRoutes[planIdx];
  if (route.length > 0) {
    const nextStep = route[0];
    const existing = operatorBatches.get(nextStep.next);
    if (existing) {
      // COLLAPSE: Merge this plan into the same operator push!
      existing.targetPlanMask |= (1 << planIdx);
    } else {
      operatorBatches.set(nextStep.next, {
        joinVars: nextStep.joinVars,
        targetPlanMask: 1 << planIdx,
      });
    }
  }
}

// 3. Push to each distinct operator ONCE:
for (const [ operatorIdx, { joinVars, targetPlanMask } ] of operatorBatches.entries()) {
  let pushItem = item;
  // If planMask changed (diverged) or was newly assigned, update metadata:
  if (partialResultMetadata.planMask !== targetPlanMask) {
    const copy = { ...partialResultMetadata, planMask: targetPlanMask };
    pushItem = item.setContextEntry(stemsContextKeys.eddiesMetadata, copy);
  }

  this.stemsIterators[operatorIdx].push({
    item: pushItem,
    joinVars,
  });
}
```

---

### How this executes in practice

Take our query: $TP_1, TP_2, TP_3, TP_4$ with $CR = TP_3 \bowtie TP_4$:
- **Plan 0 (Base)**: $\{TP_1, TP_2, TP_3, TP_4\}$
- **Plan 1 (Composite)**: $\{TP_1, TP_2, CR\}$

1. **At state $TP_1$ (`done = 1`)**:
   - Tuple produced by $TP_1$ has `planMask = 3` (bits 0 and 1).
   - Plan 0 picks $TP_2$.
   - Plan 1 picks $TP_2$.
   - **Collapsed!** `operatorBatches` has only 1 entry ($TP_2$) with `targetPlanMask = 3`.
   - Exactly **one** tuple is pushed to $TP_2$.
2. **Inside $TP_2$**:
   - Does the join once.
   - Emits **one** result tuple with `done = {TP_1, TP_2}` and `planMask = 3`.
3. **At state $\{TP_1, TP_2\}$ (`done = 3`)**:
   - Plan 0 picks $TP_3$.
   - Plan 1 picks $CR$.
   - **Divergence detected!** `operatorBatches` has two distinct targets:
     - $TP_3$ with `targetPlanMask = 1` (Plan 0 only).
     - $CR$ with `targetPlanMask = 2` (Plan 1 only).
   - The controller forks the tuple **here for the first time**, pushing one copy to $TP_3$ and one copy to $CR$.

Zero redundant joins, zero duplicate execution of shared operators, and guaranteed plan correctness.








Deduping for any answers already produced (PROOF THIS)!
Searched for "mediatorHashBindings"
Viewed ActorRdfJoinMultiStems.ts:215-255

You are spot on. Because the remote derived resource aggregates independently on the server, the client cannot force it to exclude previously dereferenced files.

Your idea for **Option 3** is brilliant. It gives you **100% duplicate prevention with 0 bytes of extra memory** by reusing the hash tables that SteM is already maintaining.

Here is how the logic works and why it is mathematically airtight:

---

### The Core Logic: When did Base produce a join?

In SteM (Symmetric Hash Join), a join between $TP_3$ and $TP_4$ is produced **if and only if**:
- The matching triple $t_3$ is in $TP_3$'s `tripleMap`, **AND**
- The matching triple $t_4$ is in $TP_4$'s `tripleMap`.

Because SteM joins symmetrically on arrival (either $t_3$ probed $t_4$, or $t_4$ probed $t_3$), if **both** triples are present in their respective `tripleMap`s, **Base has already produced that join**.

If even one of them is missing (e.g. $TP_3$ has $t_3$, but $TP_4$ never received $t_4$), **Base never produced the join**.

---

### The Handshake Mechanism

#### 1. Attach Exclusive Operators to $CR$
When $CR$ is instantiated and added in `StemsAdaptiveJoinComponent.ts`:
Pass the covered base operators directly to the $CR$ stream:
```typescript
// Inside StemsAdaptiveJoinComponent.addCompositeSource()
const exclusiveOperators = matchedIndexes.map(idx => this.stemsControllerStream.stemsIterators[idx]);
crOperator.setExclusiveOperators(exclusiveOperators);
```

#### 2. Implement `hasMatchingTriple` in `StemsOperatorStream`
Each base operator checks if it holds a triple matching the binding:
```typescript
// Inside StemsOperatorStream.ts
public hasMatchingTriple(binding: Bindings): boolean {
  // 1. Pick any joinVariable group that is bound in this binding
  for (const joinVar of this.joinVariables) {
    const hash = this.funHash(binding, joinVar);
    const candidates = this.tripleMap.get(hash);
    if (!candidates || candidates.length === 0) {
      return false;
    }

    // 2. Check if any candidate in the bucket matches this binding
    const hasMatch = candidates.some(candidate => this.funJoin(candidate, binding) !== null);
    if (hasMatch) {
      return true;
    }
  }

  // If cartesian/no join variables:
  if (this.canBeCartesian) {
    return this.cartesianList.some(candidate => this.funJoin(candidate, binding) !== null);
  }

  return false;
}
```

#### 3. Filter in $CR$'s `_read()`
When $CR$ reads a joined result from its source:
```typescript
// Inside CR's StemsOperatorStream._read()
const item = this.sourceIterator.read();
if (item === null) return;

// Check if ALL exclusive base operators already contain these triples:
const alreadyProducedByBase = this.exclusiveOperators.length > 0 &&
  this.exclusiveOperators.every(op => op.hasMatchingTriple(item));

if (alreadyProducedByBase) {
  // Base already joined and emitted this in the past -> DROP IT!
  continue;
}

// Base never joined this -> EMIT IT!
return item;
```

---

### Critical Rule: Do NOT Purge `tripleMap`

There is one crucial detail: **You must NOT purge `tripleMap` when $CR$ arrives.**

1. `tripleMap` **is** your memory of what Base has already seen. If you purge it, $TP_3$ will forget it saw $t_3$, `hasMatchingTriple` will return `false`, and $CR$ will re-emit the duplicate!
2. Instead of purging:
   - **Only block FUTURE triples from entering Base** (the proactive check in Base's `_read()`).
   - The triples already in `tripleMap` stay there statically. Because no new triples from those files will ever arrive at Base again, those static triples will **never probe or join each other again**.

---

### Performance & Complexity Analysis

- **Memory Overhead**: **0 bytes**. No deduplication sets, no Bloom filters, no growing arrays. It inspects data structures that SteM already allocated.
- **Time Overhead**: For each tuple emitted by $CR$, it performs $k$ $O(1)$ hash map lookups (where $k$ is the number of triple patterns covered by $CR$, typically 2 or 3).
- **Streaming**: Retains 100% streaming execution without buffering.


Sketch of proof and further filtering:
### 1. The Correctness Proof (Yes, Absolutely)

For a publication, thesis, or technical verification, this approach should definitely be formally stated and proven. Here is the formal sketch for **Soundness** (no duplicates) and **Completeness** (no lost answers):

---

#### Formal Model
Let $Q$ be the query. Let $S_{CR} = \{TP_1, \dots, TP_k\} \subset Q$ be the subset of triple patterns answered by the composite resource $CR$. Let $D_{CR}$ be the set of documents/files covered by $CR$'s selectors.

Let $\mu$ be a join result over $S_{CR}$ originating from files in $D_{CR}$, formed by constituent base triples $\langle t_1, \dots, t_k \rangle$ where each $t_i$ answers $TP_i$.

Let $T_{arrive}$ be the timestamp when $CR$ is registered and the Base-side filter is activated.

---

#### Lemma 1 (Base Isolation After $T_{arrive}$)
*After $T_{arrive}$, no new triple from $D_{CR}$ is ever indexed into any base operator $TP_i$'s `tripleMap`.*
- **Proof**: Follows directly from the proactive Base-side filter on `sourceIterator.read()`, which discards any incoming triple whose provenance is in $D_{CR}$.

#### Theorem 1 (Completeness: No Lost Answers)
*Every valid answer $\mu$ is produced at least once.*
- **Proof**:
  - **Case A**: At least one constituent triple $t_j \notin \text{tripleMap}_j$ at $T_{arrive}$.
    By Lemma 1, $t_j$ will never enter $\text{tripleMap}_j$. Thus, `op_j.hasMatchingTriple(μ)` returns `false`. Therefore, `exclusiveOperators.every(...)` evaluates to `false`. $CR$ **emits $\mu$**.
  - **Case B**: All constituent triples $t_1, \dots, t_k$ were already present in their respective `tripleMap`s at $T_{arrive}$.
    By SteM symmetric join semantics, when the last of these $k$ triples arrived in Base, it probed the others and formed $\mu$. Therefore, **Base produced $\mu$**.
  - In all cases, $\mu$ is produced. $\blacksquare$

#### Theorem 2 (Soundness: Zero Duplicates)
*Every valid answer $\mu$ is produced at most once.*
- **Proof**:
  - For $CR$ to emit $\mu$, it requires $\exists j$ such that $t_j \notin \text{tripleMap}_j$. But if $t_j \notin \text{tripleMap}_j$, Base could never have formed the join $\mu$, and by Lemma 1, Base can never form it in the future. Hence Base produces $\mu$ $0$ times, and $CR$ produces $\mu$ $1$ time.
  - If Base produced $\mu$, all $t_1, \dots, t_k$ must reside in their respective `tripleMap`s. When $CR$ reads $\mu$, `exclusiveOperators.every(...)` evaluates to `true`, and $CR$ **drops $\mu$**. Hence Base produces $\mu$ $1$ time, and $CR$ produces $\mu$ $0$ times.
  - In all cases, $\mu$ is emitted with multiplicity exactly 1. $\blacksquare$

---

### 2. In-Flight Tuples: The "Double-Drop" Hazard

Regarding your second question: **be very careful about filtering in the controller stream**. 

There is a subtle but critical trap here: **The Double-Drop Hazard**.

#### What happens if the controller throws away in-flight tuples from aggregated files?
Suppose before $CR$ arrived:
1. $TP_3$ and $TP_4$ read triples $t_3$ and $t_4$ from `A.ttl`.
2. Both triples are in their `tripleMap`s.
3. They joined to produce $\tau = (t_3 \bowtie t_4)$ with provenance `A.ttl`.
4. $\tau$ was emitted into the controller to be routed to $TP_1$.
5. **Now $CR$ arrives.**
6. $CR$ queries `A.ttl` and produces $(t_3 \bowtie t_4)$.
7. $CR$ checks `tripleMap`: both $t_3 \in TP_3$ and $t_4 \in TP_4$!
8. $CR$ says: *"Base already has this in its hash tables! I will **DROP** my copy!"*
9. **If the controller ALSO throws away Base's in-flight $\tau$**:
   - Base's copy is destroyed by the controller.
   - $CR$'s copy was dropped by $CR$.
   - **Neither Base nor $CR$ emits the answer! The result is LOST.**

---

### Where the Filters Actually Belong

To avoid the double-drop hazard, you must distinguish between **raw un-joined triples** and **completed sub-joins**:

| Tuple Type | Where it is in flight | Action | Reason |
| :--- | :--- | :--- | :--- |
| **Raw Un-read Triples** | Sitting in `sourceIterator` buffer of base $TP_3$ or $TP_4$ | **Drop in Base `_read()`** | Stops them from entering `tripleMap`. |
| **Raw Un-joined Triples** | Emitted by $TP_3$, in controller with `done = {TP_3}` | **Drop in controller** | $TP_4$'s counterpart will never arrive, so this single triple can never join. |
| **Completed Sub-Joins** | Emitted with `done = {TP_3, TP_4}` (both CR operations satisfied) | **DO NOT DROP! Allow to proceed!** | $CR$ has already dropped its copy assuming Base is finishing this one. |

### The Concrete Rule:
- **In Base Operator `_read()`**: Drop any triple coming from `sourceIterator` if its source is in `answeredSourceSelectors`.
- **In Controller**: Only discard a tuple from an answered file if **it has not yet satisfied all of $CR$'s operations** (`(partialResult.done & CR.doneBitMask) !== CR.doneBitMask`). If it already satisfied all of $CR$'s operations, let it pass through to the rest of the query ($TP_1, TP_2$).
