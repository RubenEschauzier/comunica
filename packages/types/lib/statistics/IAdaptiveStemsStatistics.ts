export interface IProducedTupleSummary {
  producedTuples: number[];
  producedJoinResults: number;
  producedIntermediateResults: number;
}

export interface IPlanSummary {
  fullOrders: IPlanStatistic[];
  partialOrders: IPlanStatistic[];
}
/**
 * Interface for summary statistics of the current state of execution
 */
export interface IAdaptivePlanStatistics {
  /**
   * Selectivity information on full tuple routing orders, full meaning
   * all operators are included in it
   */
  fullOrderStatistics: IPlanStatistic[];
  /**
   * Selectivity information on partial tuple routing orders
   */
  partialOrderStatistics: IPlanStatistic[];
  /**
   * Operator specific statistics
   */
  operatorSummaries: IStatsOperators[];
  /**
   * Total number of intermediate results produced at a given time
   */
  totalIntermediate: number;
  /**
   * Ratio of the number of produced results compared to intermediate results
   * showing the amount of 'extra' work is done
   */
  fanOut: number;
  /**
   * Ratio of matching triples compared to intermediate results. Showing how quickly
   * selective joins are executed
   */
  reduction: number;
  /**
   * Time it took to update routing table. This can be seen as the cost of adaptive planning
   */
  timeToUpdateRouting?: number;
}

export interface IStatsOperators {
  /**
   * Number of tuples produced by a given operator. Corresponds to number of matched bindings
   * the associated triple pattern
   */
  producedTuples: number;
  /**
   * Time the intermediate results queue was empty (s)
   */
  timeEmpty: number;
  /**
   * Time the intermediate results queue has atleast one entry (s)
   */
  timeNonEmpty: number;
  /**
   * Number of times a read() call produced no results
   */
  emptyReads: number;
  /**
   * Number of times a read() call produced a result (does not have to be a join result)
   */
  nonEmptyReads: number;
  /**
   * Number of tuples dropped by the authoritative source filter, i.e. tuples whose source
   * document a composite resource has claimed authority over
   */
  filteredByAuthoritativeSource: number;
  /**
   * Number of tuples pruned because every operator this composite resource covers had already
   * produced the matching part of them. Only ever nonzero on a composite resource operator
   */
  prunedAsAlreadyCovered: number;
}

/**
 * Interface denoting a given plan (or routing order) in the execution space and some statistics.
 */
export interface IPlanStatistic {
  order: number[];
  orderElements: Record<string, IOrderStatistic>;
  totalIntermediate: number;
  totalOutput: number;
}

/**
 * Interface denoting statistics of a given order signature.
 */
export interface IOrderStatistic {
  orderSignature: number[];
  in: number;
  out: number;
  selectivity: number;
}
