import numpy as np
from ..nsw.NSW import NSW_utility
from ...nodeManager.nodeManager.nodeCreator import Node
from ...database.DatabaseManager import DatabaseManager


class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) implementation.
    
    This class manages the multi-layer structure of HNSW graph,
    while NSW_utility handles the graph operations within each layer.
    """
    
    def __init__(self, M: int = 16, M0: int = 32, efConstruction: int = 200, 
                 mL: float = None, cache_size: int = 1000, databaseManager=None):
        """
        Initialize HNSW index.
        
        Arguments:
            M: Maximum number of connections per node (for layers > 0)
            M0: Maximum number of connections per node at layer 0 (typically 2*M)
            efConstruction: Size of dynamic candidate list during construction
            mL: Normalization factor for level generation (default: 1/ln(2))
            cache_size: Maximum vectors to keep in memory (for disk-based operation)
            databaseManager: Database manager for persistence
        """
        # Parameters (HNSW paper)
        self.M = M
        self.M0 = M0
        self.efConstruction = efConstruction
        self.mL = mL if mL is not None else 1.0 / np.log(2.0)
        
        # Graph structure
        self.nodes = {}  # {node_id: Node} - Master node registry
        self.maxLayer = -1  # Current maximum layer (starts at -1 for empty graph)
        self.entryPoint = None  # Entry point node ID at top layer
        
        # NSW utility for graph operations
        self.nsw_ops = NSW_utility(
            cache_size=cache_size,
            prefetch_size=50,
            databaseManager=databaseManager
        )
        
        # Database manager for persistence
        self.databaseManager = databaseManager if databaseManager else DatabaseManager()
    
    ####################################################################################################################
    #
    # Layer Selection Algorithm (HNSW Paper - exponential decay)
    #
    ####################################################################################################################
    def _select_layer_for_node(self) -> int:
        """
        Select layer for new node using exponential decay probability.
        
        Formula: floor(-ln(uniform(0,1)) * mL)
        This ensures higher layers have exponentially fewer nodes.
        
        Returns:
            int: Layer number (0 to potentially maxLayer+1)
        """
        return int(-np.log(np.random.uniform(0, 1)) * self.mL)
    
    ####################################################################################################################
    #
    # Node Insertion (HNSW Paper - Algorithm 1)
    #
    ####################################################################################################################
    def insert(self, newNode: Node):
        """
        Insert a new node into HNSW graph (HNSW paper Algorithm 1).
        
        Steps:
        1. Select random layer for new node (exponential distribution)
        2. Find entry point by searching from top layer down
        3. Insert node into each layer from selected layer down to 0
        4. Update entry point if new node is at higher layer
        
        Arguments:
            newNode: Node object to insert (must have vector and id)
        
        Returns:
            bool: True if insertion successful
        """
        # Step 1: Handle first node case
        if len(self.nodes) == 0:
            newNode.maxLayer = 0
            newNode.friendLists = {0: []}
            self.nodes[newNode.getId()] = newNode
            self.maxLayer = 0
            self.entryPoint = newNode.getId()
            return True
        
        # Step 2: Select layer for new node (exponential distribution)
        newNodeLayer = self._select_layer_for_node()
        newNode.maxLayer = newNodeLayer
        
        # Initialize friend lists for all layers this node will exist in
        newNode.friendLists = {layer: [] for layer in range(newNodeLayer + 1)}
        
        # Step 3: Search for nearest neighbors from top to target layer
        # Start from entry point at top layer
        ep = self.entryPoint
        
        # Phase 1: Greedy search through layers above newNodeLayer (ef=1)
        for layer in range(self.maxLayer, newNodeLayer, -1):
            # Find closest node at this layer (single nearest neighbor)
            candidates = self.nsw_ops.searchingNearestNode(
                nodes=self.nodes,
                entryPoint=ep,
                inputNode=newNode,
                layer=layer,
                ef=1
            )
            
            if candidates:
                ep = candidates[0][1].getId()  # Update entry point for next layer
        
        # Phase 2: Insert into layers from newNodeLayer down to 0
        for layer in range(min(newNodeLayer, self.maxLayer), -1, -1):
            # Determine M for this layer
            M = self.M0 if layer == 0 else self.M
            
            # Build NSW connections at this layer
            self.nsw_ops.buildNSW(
                nodes=self.nodes,
                newNode=newNode,
                layer=layer,
                M=M,
                efConstruction=self.efConstruction,
                entryPoint=ep
            )
            
            # Find best candidate for next layer's entry point
            candidates = self.nsw_ops.searchingNearestNode(
                nodes=self.nodes,
                entryPoint=ep,
                inputNode=newNode,
                layer=layer,
                ef=1
            )
            if candidates:
                ep = candidates[0][1].getId()
        
        # Step 4: Update entry point if new node is at higher layer
        if newNodeLayer > self.maxLayer:
            self.maxLayer = newNodeLayer
            self.entryPoint = newNode.getId()
        
        return True
    
    ####################################################################################################################
    #
    # K-NN Search (HNSW Paper - Algorithm 5)
    #
    ####################################################################################################################
    def search(self, queryNode: Node, k: int, ef: int = None) -> list:
        """
        Search for k nearest neighbors (HNSW paper Algorithm 5).
        
        Steps:
        1. Start from entry point at top layer
        2. Greedily traverse down to layer 0 (ef=1 per layer)
        3. At layer 0, find ef candidates
        4. Return top k results
        
        Arguments:
            queryNode: Query node (must have vector)
            k: Number of nearest neighbors to return
            ef: Size of dynamic candidate list (default: max(k, efConstruction))
        
        Returns:
            list: List of (similarity, Node) tuples (top k results)
        """
        # Handle empty graph
        if len(self.nodes) == 0 or self.entryPoint is None:
            return []
        
        # Set ef (must be >= k)
        if ef is None:
            ef = max(k, self.efConstruction)
        ef = max(ef, k)
        
        # Phase 1: Greedy search from top layer to layer 1 (ef=1)
        ep = self.entryPoint
        for layer in range(self.maxLayer, 0, -1):
            candidates = self.nsw_ops.searchingNearestNode(
                nodes=self.nodes,
                entryPoint=ep,
                inputNode=queryNode,
                layer=layer,
                ef=1
            )
            
            if candidates:
                ep = candidates[0][1].getId()
        
        # Phase 2: Search at layer 0 with ef candidates
        candidates = self.nsw_ops.searchingNearestNode(
            nodes=self.nodes,
            entryPoint=ep,
            inputNode=queryNode,
            layer=0,
            ef=ef
        )
        
        # Phase 3: Return top k results
        return candidates[:k]
    
    ####################################################################################################################
    #
    # Node Deletion (Lazy Deletion - FAISS style)
    #
    ####################################################################################################################
    def delete(self, nodeId: int):
        """
        Delete a node using lazy deletion (FAISS-style).
        
        This marks the node as deleted without modifying graph structure.
        Deleted nodes are filtered during search operations.
        
        For physical deletion with rewiring, use deleteWithRewiring().
        
        Arguments:
            nodeId: ID of node to delete
        
        Returns:
            bool: True if deletion successful
        """
        if nodeId not in self.nodes:
            return False
        
        # Mark as deleted (O(1) operation)
        self.nsw_ops.markDeleted(nodeId)
        
        # Handle entry point deletion
        if nodeId == self.entryPoint:
            # Find new entry point (first non-deleted node at highest layer)
            for layer in range(self.maxLayer, -1, -1):
                for nid, node in self.nodes.items():
                    if (not self.nsw_ops.isDeleted(nid) and 
                        hasattr(node, 'maxLayer') and 
                        node.maxLayer >= layer):
                        self.entryPoint = nid
                        return True
            
            # All nodes deleted
            self.entryPoint = None
            self.maxLayer = -1
        
        return True
    
    def deleteWithRewiring(self, nodeId: int):
        """
        Delete a node with rewiring (physical deletion).
        
        WARNING: This is O(N * layers) operation. Use sparingly.
        
        Arguments:
            nodeId: ID of node to delete
        
        Returns:
            bool: True if deletion successful
        """
        if nodeId not in self.nodes:
            return False
        
        node = self.nodes[nodeId]
        maxLayer = node.maxLayer if hasattr(node, 'maxLayer') else 0
        
        # Delete from each layer with rewiring
        for layer in range(maxLayer, -1, -1):
            M = self.M0 if layer == 0 else self.M
            self.nsw_ops.deleteNodeWithRewiring(
                nodes=self.nodes,
                deletedNodeId=nodeId,
                layer=layer,
                M=M
            )
        
        # Remove from node registry
        del self.nodes[nodeId]
        
        # Handle entry point
        if nodeId == self.entryPoint:
            # Find new entry point
            for nid, node in self.nodes.items():
                if hasattr(node, 'maxLayer'):
                    self.entryPoint = nid
                    self.maxLayer = node.maxLayer
                    break
            else:
                self.entryPoint = None
                self.maxLayer = -1
        
        return True
    
    ####################################################################################################################
    #
    # Utility Methods
    #
    ####################################################################################################################
    def getDeletionRatio(self) -> float:
        """
        Get ratio of deleted nodes (useful for rebuild decision).
        
        Returns:
            float: Deletion ratio (0.0 to 1.0)
        """
        return self.nsw_ops.getDeletionRatio(self.nodes)
    
    def shouldRebuild(self, threshold: float = 0.3) -> bool:
        """
        Check if index should be rebuilt due to high deletion ratio.
        
        Arguments:
            threshold: Deletion ratio threshold (default: 0.3 = 30%)
        
        Returns:
            bool: True if rebuild recommended
        """
        return self.getDeletionRatio() > threshold
    
    def rebuild(self):
        """
        Rebuild index by removing deleted nodes and reconstructing graph.
        
        Steps:
        1. Collect all non-deleted nodes
        2. Clear graph
        3. Re-insert all nodes
        """
        # Collect non-deleted nodes
        active_nodes = [
            node for nid, node in self.nodes.items()
            if not self.nsw_ops.isDeleted(nid)
        ]
        
        # Clear graph
        self.nodes.clear()
        self.maxLayer = -1
        self.entryPoint = None
        self.nsw_ops.clearDeletions()
        
        # Re-insert all nodes
        for node in active_nodes:
            self.insert(node)
    
    def getStats(self) -> dict:
        """
        Get statistics about the HNSW index.
        
        Returns:
            dict: Statistics including node count, layers, deletion ratio, etc.
        """
        total_nodes = len(self.nodes)
        deleted_nodes = self.nsw_ops.getDeletedCount()
        active_nodes = total_nodes - deleted_nodes
        
        # Count nodes per layer
        layer_distribution = {}
        for node in self.nodes.values():
            if hasattr(node, 'maxLayer'):
                for layer in range(node.maxLayer + 1):
                    layer_distribution[layer] = layer_distribution.get(layer, 0) + 1
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_nodes,
            'deleted_nodes': deleted_nodes,
            'deletion_ratio': self.getDeletionRatio(),
            'max_layer': self.maxLayer,
            'entry_point': self.entryPoint,
            'layer_distribution': layer_distribution,
            'parameters': {
                'M': self.M,
                'M0': self.M0,
                'efConstruction': self.efConstruction,
                'mL': self.mL
            }
        }
    
    ####################################################################################################################
    #
    # Persistence (Save/Load)
    #
    ####################################################################################################################
    def save(self, filepath: str):
        """
        Save HNSW index to disk.
        
        Arguments:
            filepath: Path to save file
        """
        # Delegate to database manager
        self.databaseManager.save_hnsw(
            filepath=filepath,
            nodes=self.nodes,
            maxLayer=self.maxLayer,
            entryPoint=self.entryPoint,
            deletedNodes=self.nsw_ops.deletedNodes,
            parameters={
                'M': self.M,
                'M0': self.M0,
                'efConstruction': self.efConstruction,
                'mL': self.mL
            }
        )
    
    def load(self, filepath: str):
        """
        Load HNSW index from disk.
        
        Arguments:
            filepath: Path to load file
        """
        # Delegate to database manager
        data = self.databaseManager.load_hnsw(filepath)
        
        self.nodes = data['nodes']
        self.maxLayer = data['maxLayer']
        self.entryPoint = data['entryPoint']
        self.nsw_ops.deletedNodes = data['deletedNodes']
        
        # Restore parameters
        params = data['parameters']
        self.M = params['M']
        self.M0 = params['M0']
        self.efConstruction = params['efConstruction']
        self.mL = params['mL']