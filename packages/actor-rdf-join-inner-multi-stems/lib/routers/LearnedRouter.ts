/**
 * 3. Query Simplification via Minimum Selectivity (Heuristic Pruning)
 * The Concept: This is a standard heuristic in SPARQL optimization (e.g., in the Jena or Virtuoso optimizers).
 * The assumption is that joins with high selectivity (filtering out many rows) should always be done early.
 *
 * Mechanism for Collapsing: States where highly selective relations have not yet been joined are pruned
 *  from the search space entirely.
 * Rule: If Relation Rselective is in the Ready set, we do not explore any state where we choose Rnon−selective first.
 *
 * Relevance to Your Bandit: You can reduce your state space by enforcing Dominance Rules.
 * If you have a Done vector where you skipped a highly selective SteM to visit a low-selectivity one,
 * do not create a Bandit for this state.
 * Hard-code the routing logic to "Always pick the highly selective SteM first" and
 * only use the Bandit for the "tied" states where the decision is non-trivial.
 *
 * This removes the "easy" decisions from your state space, leaving the Bandit to learn only the difficult trade-offs.
 *
 * Citation: Stocker, M., Seaborne, A., Bernstein, A., Kiefer, C., & Reynolds, D. (2008).
 * "SPARQL basic graph pattern optimization using selectivity estimation." WWW.
 * Rigorous Detail: They provide heuristics for ranking Triple Patterns based on selectivity,
 * effectively linearizing parts of the join tree and removing them from the DP search space.
 */
/**
 * Implements the State Aggregation strategy for Contextual Multi-Armed Bandits (UCB1).
 * This collapses the massive state space (2^N) into a small number of buckets,
 * eliminating the need for expensive matrix math while providing fast, O(1), adaptive routing.
 * * Citation (RL): Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
 * (Chapter 9: On-policy Prediction with Approximation - specifically State Aggregation).
 * Citation (Database Context): Bruno, N., & Chaudhuri, S. (2002). "Exploiting statistics on
 * query expressions for optimization." (Uses buckets to group similar sub-expressions).
 */
/**
 * The Recommended Solution: LinUCB (Linear Upper Confidence Bound)
 * To rigorously solve the "Too Many States" problem while keeping a "Tight Regret Bound,"
 * you should abandon the Tabular approach (one bandit per unique bitmask) and move to Linear Approximation.
 * Algorithm: LinUCB (Disjoint) Instead of a lookup table Bandit[DoneMask],
 * you represent the DoneMask as a feature vector x.
 * You assume the expected reward of taking action a in state x is linear: E[r]=xTθa.
 * Collapse State: Convert your Done bitmask into features:
 * x1: Number of items currently in the tuple (Depth).
 * x2: Estimated cardinality of the current tuple.
 * x3: Max selectivity of available next operators.
 *  Tight Bound: LinUCB has a rigorous regret bound of O~(T).
 * Citation for LinUCB: Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010).
 *  "A contextual-bandit approach to personalized news article recommendation." WWW.
 * This is the standard, rigorous way to solve the curse of dimensionality in Bandit problems.
 */
