export class NodeStateSpace {
    /** Node denoting either a join (parent node) or a binding stream (leaf)
     */
    public featureArray: Array<number>;
    // public parent: NodeStateSpace;
    public parentId: number;
    public children: NodeStateSpace[];
    /* Debug Only */
    public childrenId:  number[]
    public childrenIdString: string;
    /** Node identifier */ 
    public id: number;
    public depth: number;
    /** Boolean denoting whether a node is already part of a join */
    public joined: boolean;
    /** Temporary cardinality figure, it should be an array */
    public cardinality: number;

    constructor(idNode: number, cardinality: number) {
        // Temporary cardinality
        this.cardinality = cardinality;
        this.children = [];
        this.childrenId = [];
        this.id = idNode;
        this.joined = false;
    }

    public setChild(childNode: NodeStateSpace){
        console.assert(this.children.length < 2, {numberChildren: this.children.length,
            errorMsg: "Cannot add more than 2 children"})
        this.children.push(childNode);
        this.childrenId.push(childNode.id);
        this.childrenIdString = JSON.stringify(this.childrenId);
    }

    public setParent(parentNode: NodeStateSpace){
        // this.parent = parentNode;
        this.parentId = parentNode.id;
    }

    public setJoined(value: boolean){
        this.joined=value;
    }

    public setFeatures(features: number[]){
        this.featureArray = features;
    }

    public isLeave(){
        return this.children.length==0
    }

    public setDepth(depth: number){
        this.depth=depth;
    }

    public setFeature(featureArray: number[]){
        this.featureArray = featureArray;
    }
}

export class StateSpaceTree{
    /** Tree of nodes denoting (partial) join tree. Trees are constructed from bottom up,
     * where the bottom of the tree represents the unjoined binding streams*/
    root: NodeStateSpace;
    numNodes: number;
    nodesArray: NodeStateSpace[];
    adjacencyMatrix: number[][];
    numLeaveNodes: number;
    numUnjoined: number;
    constructor() {
        this.numNodes = 0;
        this.numUnjoined = 0;
        this.numLeaveNodes = 0;
        this.nodesArray = [];
        this.adjacencyMatrix = [[]];
    }
    public addLeave(leave: NodeStateSpace){
        this.nodesArray.push(leave);
        this.incrementNumNodes();
        this.numUnjoined += 1;
        /* Expand the adjacency matrix by adding column and row with zeros */
        if (this.numNodes>1){
            this.adjacencyMatrix.push(Array(this.numNodes-1).fill(0));
        }
        for (let i=0; i<this.numNodes; i++){
            /* Preserve self connections in tree*/
            i == this.numNodes-1 ? this.adjacencyMatrix[i].push(1) : this.adjacencyMatrix[i].push(0);
            // if(i==this.numNodes-1){
            //     this.adjacencyMatrix[i].push(1);
            // }
            // else{
            //     this.adjacencyMatrix[i].push(0);
            // }
        }
    }

    public addParent(leaveIndexes: number[], parent: NodeStateSpace){
        // Bad code, should do in for loop
        console.assert (leaveIndexes.length == 2, {numberChildren: leaveIndexes.length,
            errorMsg: "Can only supply 2 indices to add, no more or less"});
        const node1: NodeStateSpace = this.nodesArray[leaveIndexes[0]];
        const node2: NodeStateSpace = this.nodesArray[leaveIndexes[1]];

        // Set child and parent relations
        
        node1.setParent(parent); node2.setParent(parent);
        parent.setChild(node1); parent.setChild(node2);
        // Set joined status to true for each of the two child nodes
        node1.setJoined(true); node2.setJoined(true);
        this.numUnjoined += -1;

        // The depth of the parent node is the deepest of the two nodes joined + 1
        // To represent the shape of the join tree
        const parentDepth: number = Math.max(node1.depth, node2.depth) + 1;
        parent.setDepth(parentDepth);

        // Update child nodes in the array and add parent node to the array
        this.nodesArray[leaveIndexes[0]] = node1; this.nodesArray[leaveIndexes[1]] = node2;
        this.nodesArray.push(parent);

        /* Add entry to adjacency matrix*/
        for (let i = 0; i<this.numNodes; i++){
            this.adjacencyMatrix[i].push(0);
        }

        this.adjacencyMatrix.push(Array(this.numNodes+1).fill(0));
        this.adjacencyMatrix[this.numNodes][this.numNodes] = 1;
        this.adjacencyMatrix[this.numNodes][leaveIndexes[0]] = 1;
        this.adjacencyMatrix[this.numNodes][leaveIndexes[1]] = 1;

        this.incrementNumNodes();
    }
    public removeLastAdded(){
        /*Remove the last added node, in this we do not remove the parents, however by indicating joined = false we know 
          The node has no parents        
        */
        if (this.nodesArray){
            const lastNode: NodeStateSpace = this.nodesArray.pop()!;
            const removedNodeChildren: NodeStateSpace[] = lastNode.children;

            for (const child of removedNodeChildren){
                this.nodesArray[child.id].setJoined(false);
            }

            /* Remove the parent node from the adjacency matrix*/
            this.adjacencyMatrix.pop();
            for (let i = 0; i<this.numNodes-1; i++){
                this.adjacencyMatrix[i].pop();
            }
            this.decrementNumNodes();
            this.numUnjoined += 1;
        }
        else{
            throw new Error(`This tree is empty`);
        }
    }
    public setParentFeatures(parent: NodeStateSpace){
        // Insert logic for features, not sure what do here yet, prob some averaging
    }
    public calculateRoot(){
        // We calculate the root at the end of creating of the tree, this takes O(N), where N is the number of nodes
        // Due to us making the tree higher from bottom down this can't be relegated to the addParent function and
        // By separating the logic we only have to call once.
        let rootDepth: number = 0;
        let rootNode: NodeStateSpace = this.nodesArray[0];
        for (let i = 0; i<this.nodesArray.length; i++){
            // Iterate over all nodes and find the deepest node
            if (this.nodesArray[i].depth > rootDepth){
                rootDepth = this.nodesArray[i].depth;
                rootNode = this.nodesArray[i];
            }
        }
        // Set root node to deepest node in the tree
        this.setRoot(rootNode)
    }
    public emptyStateSpace(){
        this.numNodes = 0;
        this.numUnjoined = 0;
        this.numLeaveNodes = 0;
        this.nodesArray = [];
        this.adjacencyMatrix = [[]];
    }

    public isEmpty(){
        return (this.numNodes==0 && this.nodesArray.length==0);
    }
    public setNumLeaveNodes(leaves: number){
        this.numLeaveNodes = leaves;
    }
    private setRoot(rootNode: NodeStateSpace){
        this.root = rootNode;
    }
    private incrementNumNodes(){
        this.numNodes += 1;
    }
    private decrementNumNodes(){
        this.numNodes += -1;
    }

}
// // Some test code making a tree and joining it
// let joinTree: StateSpaceTree = new StateSpaceTree()
// for(let i = 0; i<5; i++){
//     let newNode: NodeStateSpace = new NodeStateSpace(i);
//     newNode.setDepth(0);
//     joinTree.addLeave(newNode);
// }
// let allJoined: boolean = false;
// let nodeIndex: number = 4;
// while (!allJoined){
//     let numUnjoined: number = 0;
//     const toJoin: NodeStateSpace[] = [];
//     for (let j=0;j<joinTree.nodesArray.length;j++){
//         if (!joinTree.nodesArray[j].joined){
//             toJoin.push(joinTree.nodesArray[j]);
//             numUnjoined += 1;
//         }
//         if (numUnjoined >= 2) {
//             break;
//         }
//     }

//     if (numUnjoined == 1){
//         allJoined = true;
//         break;
//     }
//     nodeIndex+=1;
//     let parentNew = new NodeStateSpace(nodeIndex);
//     joinTree.addParent([toJoin[0].id, toJoin[1].id], parentNew);

// }
// console.log(joinTree)