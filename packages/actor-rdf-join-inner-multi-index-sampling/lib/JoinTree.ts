export class JoinTree{

    public entries: Set<number>;
    public cost: number;
    public joinOrder: Set<number>[];

    public constructor(
        joinOrder: Set<number>[],
        entries: Set<number>
    ){
        this.joinOrder = joinOrder;
        this.entries = entries;
        this.cost = Number.NEGATIVE_INFINITY;
    }

    /**
     * Merge join trees (TODO Consider what order the operator should be used in. Include in cost function?).
     * The trees should be for disjoint entries
     * @param tree2 (Optimal) tree to merge
     */
    public static mergeTree(tree1: JoinTree, tree2: JoinTree){
        const entriesMerged = new Set([...tree1.entries, ...tree2.entries]);
        const joinOrderMerged = [...tree1.joinOrder, ...tree2.joinOrder, entriesMerged];
        return new JoinTree(joinOrderMerged, entriesMerged);
    }

    public setCost(cost: number){
        this.cost = cost;
    }   
}