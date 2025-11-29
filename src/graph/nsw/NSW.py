#set the number of edge to have 
#edge management
    #find closest Node -> count -> if there is space ,add-> 
from ...nodeManager.nodeManager.nodeCreator import Node
import numpy as np
from ...nodeManager.sortManager.priorityQueue.priorityQueue import PriorityQueue
from ...nodeManager.sortManager.heapSort.heap import HeapSorter
from queue import Queue
from collections import deque


class NSW_utility:
    """Utility class for NSW graph operations (stateless, no graph storage)"""
    def __init__(self, cache_size:int=1000, prefetch_size:int=50, databaseManager=None) -> None:
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        self.databaseManager = databaseManager if databaseManager else DatabaseManager()
        
        # Lazy Deletion (FAISS-style) - O(1) lookup with set
        self.deletedNodes = set()
        
        # Cache Management
        self.loaded_node_ids = deque() # FIFO queue for eviction


    ####################################################################################################################
    #
    # Node Searching algorithm 
    #
    ####################################################################################################################
    def searchingNearestNode(self, nodes: dict, entryPoint: int, inputNode: Node, layer: int, ef: int = 1) -> list:
        """
        Find the local minimum (candidate of closest nodes in NSW) and return metadata.
        
        Arguments:
            nodes: Dictionary of all nodes {id: Node}
            entryPoint: The id of entry point
            inputNode: Subject of comparison node
            layer: Which layer to search in
            ef: Size of candidate list

        Returns: list of (similarity, Node) tuples
        """
        # nns is the priority queue
        nns = PriorityQueue(maxSize=ef)
        
        # Case 1: If there are no existing nodes
        if len(nodes) == 0:
            return nns.heap
        
        # Initialize the closest node
        closestNode = nodes[entryPoint]
        
        # Skip if entry point is deleted (lazy deletion check)
        if entryPoint in self.deletedNodes:
            # Find first non-deleted node
            found = False
            for nodeId, node in nodes.items():
                if nodeId not in self.deletedNodes:
                    closestNode = node
                    found = True
                    break
            
            if not found:
                return nns.heap  # All nodes deleted
        
        # PREFETCH: Ensure entry point is loaded
        self.prefetch_neighborhood(closestNode, layer)
        
        similarity = self.get_distance(inputNode, closestNode)
        nns.insert(value=(similarity, closestNode))
        
        while True:
            # PREFETCH: Load all neighbors of the current closest node in one batch
            self.prefetch_neighborhood(closestNode, layer)

            # Step 2: Flag to track if a closer node is found
            renewalInfo = False

            # Step 3: Iterate through the neighbors (layer-specific friend list)
            # Access layer-specific friendList or fallback to single friendList
            friendList = closestNode.friendLists.get(layer, []) if hasattr(closestNode, 'friendLists') else closestNode.friendList
            for _, neighborNode in friendList:
                # Skip deleted nodes (lazy deletion filter - FAISS-style)
                if neighborNode.id in self.deletedNodes:
                    continue
                
                # Calculate the similarity to the neighbor
                # Note: neighborNode.vector should be loaded now due to prefetch_neighborhood
                similarity = self.get_distance(inputNode, neighborNode)               

                # if size of nns is less than ef, then add 
                if nns.getSize() < ef:
                    nns.insert(value=(similarity, neighborNode))
                    renewalInfo = True
                
                # If a closer node is found, update the closest node and similarity
                else:
                    # Compare with the worst candidate in the queue (tail)
                    # We want to maximize similarity, so if new > worst, we replace.
                    if similarity > nns.tail()[0]:
                        # replace 
                        nns.repalceTail(newTuple=(similarity, neighborNode))
                        # implement heap sort
                        nns.sortArray()
                        # turn flag True
                        renewalInfo = True
            
            # Base case: when no addition to nns (no neighbor was better than what we have)
            if not renewalInfo:
                return nns.heap
            
            # Update closestNode to the best candidate found in nns for the next iteration
            # nns.heap[0] is the best candidate (highest similarity)
            best_candidate = nns.heap[0][1] 
            
            # If the best candidate is the same as where we are, we are at a local maximum
            if best_candidate != closestNode:
                closestNode = best_candidate
            else:
                return nns.heap

    ####################################################################################################################
    #
    # NSW Construction algorithm 
    #
    ####################################################################################################################
    def buildNSW(self, nodes: dict, newNode: Node, layer: int, M: int, efConstruction: int, entryPoint: int = None):
        """
        NSW construction (HNSW paper compliant).
        
        Arguments:
            nodes: Dictionary of all nodes {id: Node}
            newNode: The node to insert into the NSW graph
            layer: Which layer to build connections in
            M: Maximum number of connections per node
            efConstruction: Size of candidate list for construction
            entryPoint: The starting node ID for search (if None, uses a random node)
        
        Returns:
            bool: True if insertion successful, False otherwise
        """
        # Step 1: Handle first node case
        if len(nodes) == 0:
            nodes[newNode.getId()] = newNode
            return True
        
        # Step 2: Set default entry point
        if entryPoint is None:
            entryPoint = list(nodes.keys())[0]
        
        # Step 3: Find nearest neighbors (ef candidates)
        candidates = self.searchingNearestNode(
            nodes=nodes,
            entryPoint=entryPoint,
            inputNode=newNode,
            layer=layer,
            ef=efConstruction
        )
        
        # Step 4: Select M neighbors from candidates (pruning)
        selectedNeighbors = self.selectNeighbors(newNode, candidates, M)
        
        # Step 5: Add new node to graph
        nodes[newNode.getId()] = newNode
        
        # Step 6: Add bidirectional links
        for similarity, neighborNode in selectedNeighbors:
            # Add neighbor to newNode's friend list at this layer
            self._addToFriendList(newNode, neighborNode, similarity, layer)
            
            # Add newNode to neighbor's friend list with pruning
            self.connectNodes(neighborNode, newNode, layer, M, similarity)
        
        return True
    
    def _addToFriendList(self, node: Node, targetNode: Node, similarity: float, layer: int):
        """
        Helper: Add targetNode to node's friend list at specified layer.
        
        Arguments:
            node: Node whose friend list will be updated
            targetNode: Node to add
            similarity: Similarity score
            layer: Which layer to add to
        """
        if hasattr(node, 'friendLists'):
            # Multi-layer node structure
            if layer not in node.friendLists:
                node.friendLists[layer] = []
            node.friendLists[layer].append((similarity, targetNode))
        else:
            # Single-layer node structure (backward compatibility)
            node.friendList.append((similarity, targetNode))
    
    def selectNeighbors(self, queryNode: Node, candidates: list, M: int) -> list:
        """
        Select M neighbors from candidates (HNSW paper Algorithm 4 - simple heuristic).
        
        Arguments:
            queryNode: The query node
            candidates: List of (similarity, node) tuples
            M: Number of neighbors to select
        
        Returns:
            List of selected (similarity, node) tuples (at most M items)
        """
        # Simple heuristic: select top M by similarity
        # For more advanced pruning, implement diversity-based selection
        sortedCandidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        return sortedCandidates[:M]
    
    def connectNodes(self, node1: Node, node2: Node, layer: int, Mmax: int, similarity: float = None):
        """
        Add edge from node1 to node2 with pruning (HNSW paper compliant).
        
        Arguments:
            node1: Node to add edge from
            node2: Node to connect to
            layer: Which layer to add connection in
            Mmax: Maximum connections for node1
            similarity: Precomputed similarity (if None, will calculate)
        
        Returns:
            None
        """
        # Calculate similarity if not provided
        if similarity is None:
            similarity = self.get_distance(node1, node2)
        
        # Add node2 to node1's friend list at specified layer
        self._addToFriendList(node1, node2, similarity, layer)
        
        # Get current friendList for this layer
        friendList = node1.friendLists.get(layer, []) if hasattr(node1, 'friendLists') else node1.friendList
        
        # If exceeds Mmax, prune to keep best M connections
        if len(friendList) > Mmax:
            # Prune: select best Mmax neighbors from current friends
            pruned = self.selectNeighbors(node1, friendList, Mmax)
            if hasattr(node1, 'friendLists'):
                node1.friendLists[layer] = pruned
            else:
                node1.friendList = pruned
    
    ####################################################################################################################
    #
    # Cache and Distance Management
    #
    ####################################################################################################################
    def _ensure_vector_loaded(self, node: Node):
        """
        Ensures the node has its vector loaded. If not, triggers prefetching 
        of this node and its neighbors (Topology-based Prefetching).
        """
        if node.vector is not None:
            return

        # Strategy: BFS to find relevant nodes (self + neighbors) to fill the batch
        ids_to_fetch = []
        nodes_to_update = []
        
        queue = deque([node])
        visited = {node.id}
        
        while queue and len(ids_to_fetch) < self.prefetch_size:
            curr = queue.popleft()
            
            # If vector is missing, mark for fetching
            if curr.vector is None:
                ids_to_fetch.append(curr.id)
                nodes_to_update.append(curr)
            
            # Add friends to queue to prefetch their vectors too
            # Try layer 0 first, fallback to friendList
            friendList = curr.friendLists.get(0, []) if hasattr(curr, 'friendLists') else curr.friendList
            for _, friend in friendList:
                if friend.id not in visited:
                    visited.add(friend.id)
                    queue.append(friend)
        
        # Fetch batch from DB
        if ids_to_fetch:
            vectors_map = self.databaseManager.get_vectors(ids_to_fetch)
            
            for node_obj in nodes_to_update:
                if node_obj.id in vectors_map:
                    node_obj.vector = vectors_map[node_obj.id]
                    self.loaded_node_ids.append(node_obj.id)

    def _manage_cache(self, nodes: dict):
        """
        Evicts old vectors if cache exceeds size.
        
        Arguments:
            nodes: Dictionary of all nodes {id: Node}
        """
        while len(self.loaded_node_ids) > self.cache_size:
            evict_id = self.loaded_node_ids.popleft()
            # Only unload if it's still in the graph
            if evict_id in nodes:
                nodes[evict_id].vector = None

    def get_distance(self, node1: Node, node2: Node) -> float:
        """
        Calculates distance, ensuring vectors are loaded from disk if necessary.
        """
        self._ensure_vector_loaded(node1)
        self._ensure_vector_loaded(node2)
        return self.getCosine_similarity(node1.vector, node2.vector)

    def prefetch_neighborhood(self, center_node: Node, layer: int = 0):
        """
        Optimized Prefetching:
        Fetches the center node and ALL its direct neighbors in one batch 
        BEFORE we start calculating distances.
        
        Arguments:
            center_node: The node whose neighborhood to prefetch
            layer: Which layer's friendList to prefetch from
        """
        ids_to_fetch = []
        nodes_to_update = []

        # 1. Check center node
        if center_node.vector is None:
            ids_to_fetch.append(center_node.id)
            nodes_to_update.append(center_node)

        # 2. Check all neighbors (layer-aware)
        friendList = center_node.friendLists.get(layer, []) if hasattr(center_node, 'friendLists') else center_node.friendList
        for _, neighbor in friendList:
            if neighbor.vector is None:
                ids_to_fetch.append(neighbor.id)
                nodes_to_update.append(neighbor)
            
            # Memory Safety: Stop if we exceed prefetch limit
            if len(ids_to_fetch) >= self.prefetch_size:
                break
        
        # 3. Batch Fetch
        if ids_to_fetch:
            vectors_map = self.databaseManager.get_vectors(ids_to_fetch)
            for node_obj in nodes_to_update:
                if node_obj.id in vectors_map:
                    node_obj.vector = vectors_map[node_obj.id]
                    self.loaded_node_ids.append(node_obj.id)

    def getCosine_similarity(self, vector1st: np.ndarray, vector2nd: np.ndarray):
        """
        Calculate the cosine similarity between two vectors.
        
        Arguments:
            vector1st: The first vector (e.g., embedding vector of a node).
            vector2nd: The second vector (e.g., embedding vector of another node).
        
        Returns:
            float: The cosine similarity between the two vectors.
        """
        # Calculate the dot product of A and B
        dot_product = np.dot(vector1st, vector2nd)
        
        # Calculate the magnitudes (norms) of A and B
        normOfVector1st = np.linalg.norm(vector1st)
        normOfVector2nd = np.linalg.norm(vector2nd)
        
        # Compute the cosine similarity
        cosine_sim = dot_product / (normOfVector1st * normOfVector2nd)
        
        return cosine_sim

    ####################################################################################################################
    #
    # Lazy Deletion (FAISS-style) - Simple and Fast
    #
    ####################################################################################################################
    def markDeleted(self, nodeId: int):
        """
        Mark a node as deleted (O(1) operation).
        
        This uses lazy deletion strategy (FAISS-style):
        - Fast: O(1) time complexity
        - No graph rewiring needed
        - Deleted nodes are filtered during search
        - Applies to ALL layers
        
        Arguments:
            nodeId: ID of node to mark as deleted
        
        Returns:
            bool: True if marked successfully
        """
        self.deletedNodes.add(nodeId)
        return True
    
    def unmarkDeleted(self, nodeId: int):
        """
        Unmark a previously deleted node (restore it).
        
        Arguments:
            nodeId: ID of node to restore
        
        Returns:
            bool: True if unmarked successfully, False if not found
        """
        if nodeId in self.deletedNodes:
            self.deletedNodes.discard(nodeId)
            return True
        return False
    
    def isDeleted(self, nodeId: int) -> bool:
        """
        Check if a node is marked as deleted.
        
        Arguments:
            nodeId: ID of node to check
        
        Returns:
            bool: True if node is marked as deleted
        """
        return nodeId in self.deletedNodes
    
    def getDeletionRatio(self, nodes: dict) -> float:
        """
        Calculate the ratio of deleted nodes.
        Useful for deciding when to rebuild the index.
        
        Arguments:
            nodes: Dictionary of all nodes {id: Node}
        
        Returns:
            float: Ratio of deleted nodes (0.0 to 1.0)
        """
        if len(nodes) == 0:
            return 0.0
        return len(self.deletedNodes) / len(nodes)
    
    def clearDeletions(self):
        """
        Clear all deletion marks. Typically called after rebuilding the index.
        """
        self.deletedNodes.clear()
    
    def getDeletedCount(self) -> int:
        """
        Get the number of deleted nodes.
        
        Returns:
            int: Number of deleted nodes
        """
        return len(self.deletedNodes)

    ####################################################################################################################
    #
    # Deleting node with Rewiring (Physical Deletion - O(N) operation)
    #
    ####################################################################################################################
    def deleteNodeWithRewiring(self, nodes: dict, deletedNodeId: int, layer: int, M: int):
        """
        Delete node with rewiring strategy (maintains graph connectivity).
        
        WARNING: This is O(N) operation. For frequent deletions, use markDeleted() instead.
        
        Strategy:
        1. Identify 'incoming nodes' (nodes that point to the deleted node).
           - Since HNSW is a directed graph, we must scan all nodes to find these.
        2. Identify 'outgoing neighbors' (nodes the deleted node points to).
        3. Rewire: Connect incoming nodes to outgoing neighbors to bridge the gap.
           (X -> Deleted -> Y  becomes  X -> Y)
        4. Remove deleted node from graph.
        
        Arguments:
            nodes: Dictionary of all nodes {id: Node}
            deletedNodeId: ID of node to delete
            layer: Which layer to delete from
            M: Maximum connections per node
        
        Returns:
            bool: True if deletion successful
        """
        # Step 1: Get the node to delete
        if deletedNodeId not in nodes:
            return False
        
        deletedNode = nodes[deletedNodeId]
        
        # Step 2: Get outgoing neighbors (candidates for rewiring)
        # These are Y in (X -> Deleted -> Y)
        outgoing_neighbors = []
        deleted_friend_list = deletedNode.friendLists.get(layer, []) if hasattr(deletedNode, 'friendLists') else []
        for _, neighbor in deleted_friend_list:
            outgoing_neighbors.append(neighbor)
            
        # Step 3: Scan ALL nodes to find incoming edges (X -> Deleted)
        # This is O(N) because we don't have a reverse index
        for nodeId, node in nodes.items():
            if nodeId == deletedNodeId:
                continue
                
            # Access friend list safely
            if hasattr(node, 'friendLists'):
                if layer not in node.friendLists:
                    continue
                friendList = node.friendLists[layer]
            else:
                friendList = node.friendList
            
            # Check if this node connects to deletedNode
            is_connected = False
            new_friend_list = []
            
            for sim, friend in friendList:
                if friend.id == deletedNodeId:
                    is_connected = True
                    # Do not add deleted node to new list
                else:
                    new_friend_list.append((sim, friend))
            
            # If connection found, update and rewire
            if is_connected:
                # 3a. Update friend list (remove deleted node)
                if hasattr(node, 'friendLists'):
                    node.friendLists[layer] = new_friend_list
                else:
                    node.friendList = new_friend_list
                
                # 3b. Rewire: Try to connect this node (X) to deleted node's neighbors (Y)
                for neighbor_y in outgoing_neighbors:
                    # Avoid self-loops and duplicates (connectNodes handles duplicates usually, but good to check)
                    if neighbor_y.id == node.id:
                        continue
                        
                    # Calculate distance and try to connect
                    # connectNodes will handle pruning if list exceeds M
                    self.connectNodes(node, neighbor_y, layer, M, similarity=None)

        # Step 4: Remove deleted node from graph
        # Note: We do NOT delete the node from the 'nodes' dictionary here.
        # This utility class only handles rewiring. The actual removal of the node object
        # from the global storage should be handled by the caller (HNSW manager)
        # after ensuring it is removed from all layers.
        # del nodes[deletedNodeId]  <-- REMOVED
        
        return True
