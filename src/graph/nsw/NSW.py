#set the number of edge to have 
#edge management
    #find closest Node -> count -> if there is space ,add-> 
from ...nodeManager.nodeManager.nodeCreator import Node
import numpy as np
from ...nodeManager.sortManager.priorityQueue.priorityQueue import PriorityQueue
from ...nodeManager.sortManager.heapSort.heap import HeapSorter
from queue import Queue
from collections import deque


#Create graph
#Traverse graph
    #

class NSW:
    """each element is """
    def __init__(self,maximumFriend:int,maxSize:int) -> None:
        self.maximumFriend = maximumFriend
        self.maxSize = maxSize
        self.nodeIdes = {}    #{id:id}
        self.nodes = {}   #{id:Node}      <= ここいらない代わりにnodeに情報を持たす
        self.priorityQueue = PriorityQueue(maxSize)
        self.databaseManager = DatabaseManager()

    #<<Creating graph>>-----------------------------------------
    def createNewGraph(self):
        """
        Take charge of creating new graph
        """


    #<
    def insertNode(self,entryNode:Node,entryPoint:int):
        """
        Brief:
            Inserts a new node into the graph. It first checks if there is no predecessor, in which case the node is added directly.
            If a predecessor exists, it searches for the nearest node and attempts to create an edge. If an edge cannot be created, 
            it incrementally increases the "friend" search range and tries again until successful or the maximum threshold is reached.

        Arguments:
            entryNode (Node): The node to be inserted into the graph.
            entryPoint (int): The point in the graph (or existing node) to begin the search for the nearest neighbor.

        Returns:
            bool: True if the node was successfully inserted (i.e., an edge was created), False otherwise.
        """
        #<<Case1:No predecessor>>
        if len(self.nodes) == None:
            self.nodes[entryNode.getId()] = entryNode
            return True
        

        #<<Case2:With Predecessor>>
        else:
            #<find candidate of nearest Node>
            NearestNodeList:tuple = self.searchingNearestNode(entryPoint=entryPoint,inputNode=entryNode)
            #If possible to create edge
            if self.createEdge(subjectNode=NearestNodeList[1],inputNode=entryNode):
                return True

            
            #if cannot create edge, go to next closer node and do same for until creating edge(stop condition is do all of node)
            else:              
                # if cannot 
                    #try to expand ef by one and try to find and if reach the num more than length of this graph -> raise error ?
                for i in range(2,self.maximumFriend):
                    subjectNode = self.searchingNearestNode(entryPoint=entryNode,inputNode=entryNode,ef=i)[i]
                    if self.createEdge(subjectNode=subjectNode,inputNode=entryNode):
                        return True
                #worst case: cannot create edge
                return False

    def createEdge(self,subjectNode,inputNode,friendListSize)->bool:
        """
        Attempts to create an edge between subjectNode and inputNode by adding inputNode to
        subjectNode's friend list. If the friend list is full, it will replace the least 
        similar node if inputNode is more similar.

        Args:
            subjectNode: The node to which the edge is being added (subject node).
            inputNode: The node being added to the friend list of subjectNode.
            friendListSize: The maximum number of friends allowed in subjectNode's friend list.
        
        Returns:
            bool: 
                - True if the edge was successfully created (i.e., inputNode added to friend list).
                - False if the edge creation failed (i.e., inputNode wasn't similar enough).
        """
        similarity = self.getCosine_similarity(vector1st=subjectNode,vector2nd=inputNode)
        #if friend node has space
        if len(subjectNode.friendList)  < friendListSize:
            #similarity
            subjectNode.friendList.append((similarity,inputNode))
            HeapSorter.heapsort(subjectNode.friendList)
            return True
        #else,compreing 
        else:
            #if input is closer than farthest node
            if subjectNode.friendList[-1][0] < similarity:
                #replce
                subjectNode.friendList[-1] = (similarity,inputNode)
                HeapSorter.heapsort(subjectNode.friendList)
                return True
            else:
                #fail to creating edge
                return False

    #<<Searching node from NSW before being disk based>>-------------------------------------------------------------
    def searchingNearestNode(self, entryPoint: int, inputNode: Node,ef=1) ->tuple:
        """
        Find the local minimum(candidate of closest node in NWW) and return meta data 
        Arguent:
            entryPoint:the id of entry point
            inputNode:subject of comparison node

        Returns:tuple (similarity,Node)
        """
        #nns is the priority queue
        nns = PriorityQueue(maxSize=ef)
        
        # Case 1: If there are no existing nodes
        if len(self.nodes) == 0:
            return nns.heap
        
        # Initialize the closest node
        closestNode = self.nodes[entryPoint]
        similarity = self.calculateDistance(inputNode,closestNode)
        self.priorityQueue.insert(value=(similarity,closestNode))
        self.priorityQueue()
        
        while True:
            # Step 1: Calculate the similarity to the current closest node
            closestsimilarity = self.calculateDistance(inputNode, closestNode)
            
            # Step 2: Flag to track if a closer node is found
            renewalInfo = False

            # Step 3: Iterate through the neighbors (friend list)
            for neighborNode in closestNode.friendList:
                # Calculate the similarity to the neighbor
                similarity = self.calculateDistance(inputNode, neighborNode)
                

                #if seze of nns is less than ef,then add 
                if nns.getSize() < ef:
                    nns.insert(value=(similarity,neighborNode))
                    renewalInfo = True
                
                # If a closer node is found, update the closest node and similarity
                else:
                    if similarity < nns.tail():
                        #replace 
                        nns.repalceTail(newTuple=(similarity,neighborNode)) = (similarity,neighborNode)
                        #implement heap sort
                        nns.sortArray()
                        #turn flag True
                        renewalInfo = True
            
                # Base case: when noaddition to nns(father than farthest node in nns)
                if not renewalInfo:
                    return nns.heap
    
    
    #<<Deleting node from NSW>>
    def deleteNode(self, deletingNode):
        """
        Deletes a node from the graph, removes it from other nodes' friend lists, and 
        reconstructs the graph's connectivity.

        Args:
            deletingNode: The node object to be deleted.
            
        Returns:
            None
        """
        #Traverse to find the node which has deleted node at friend listand create 

        #iterate through to find mnn and try to create node between



        
        if deletingNode.id in self.nodes:
            del self.nodes[deletingNode.id]  # Remove the node from the graph
            # Remove the node from other nodes' friend lists
            self.deleteNodeFromFriendList(deletingNode)
            # Rebuild connectivity for the affected nodes after deletion
            self.reconstructionConnectivity(deletingNode)



    def deleteNodeFromFriendList(self,deletingNode):
        """
        Removes the given node from all other nodes' friend lists.

        Args:
            deletingNode: The node to be removed from other nodes' friend lists.
            
        Returns:
            None
        """
        




        for node in self.nodes.values():
            # Filter out the deletingNode from the friend list of each node
            node.friendList = [(similarity, friend) for similarity, friend in node.friendList if friend != deletingNode]

    def deletingNode(self,deletedNode:Node):
        """
        Brief:

        Argument:

        Returns:

        """
        #Step1:get the friend node of deleted node and save
        temp = deletedNode.friendList
        #step2:delete node
        #delete from nsw
        del self.nodes[deletedNode.idToRefference]
        #delete from friend list from node in NSW and get the node which has it
        nodePossesingDeletedNode = self.deleteNodeFromFriend(self,deletedNode)
        #step3:rewireing 
    
    def deleteNodeFromFriendList(self, deletedNode: Node) -> list:
        """
        Brief:
        This method removes a specified node (`deletedNode`) from the `friendList` of all other nodes 
        in the network. It returns a list of nodes from which the `deletedNode` was removed.

        Params:
        deletedNode: Node - The node to be removed from the friend lists.

        Returns:
        list: A list of nodes that had the `deletedNode` in their `friendList`.
        """
        result = []

        # Iterate over all nodes
        for node in self.nodes:
            # Iterate through the friendList of each node
            for index, metaData in enumerate(node.friendList):  # Iterate through friendList properly
                if metaData[1] == deletedNode:  # Compare node's friend with deletedNode
                    del node.friendList[index]  # Remove the friend from the list
                    result.append(node)  # Add node to the result list
                    break  # Break out of the inner loop after removing the friend

        return result

    
        

     
    def rewireEdge(self,parentNodes:list,friendListOfDeletedNode:list):
        nn = friendListOfDeletedNode[0][1]
        subOrdinates = friendListOfDeletedNode.remove(nn)
        #Step1:let nns to replace position of deleted node
        for parentNode in parentNodes:
            #add nns to friendList
            #get distance
            cosineSimilarity = self.getCosine_similarity(vector1st=parentNode.embeddingVector,vector2nd=nn.embeddingVector)
            parentNode.friendList.append((cosineSimilarity,nn))
            HeapSorter.heapsort(parentNode.friendList)

        #step2:rewire subordinate node of deleted node

    def handleRemainNode(self,objectiveNode:Node,remainFriends:list):
        """
        Brief:
        Params:
        Returns:
        """
        #iterate through ojectiveNode untill all of node is fitted in
        for remainFriend in remainFriends:
            #find the friend list which accept
            self.handleRemainNodeHelper(objectiveNode=objectiveNode,subjectiveNode=remainFriend[1])
        return True

        

        

    def handleRemainNodeHelper(self,objectiveNode:Node,subjectiveNode:Node)->bool:
        """
        Brief:
        Add or update the subjectiveNode in the graph structure based on similarity.
        
        Params:
        - objectiveNode: Node - The starting node for the BFS.
        - subjectiveNode: Node - The node to be added or updated.
        
        Returns:
        - bool: True if the subjectiveNode was successfully added/updated, False otherwise.
        """
        similarity, startingNode = objectiveNode
        queue = [startingNode]

        while queue:
            currentNode = queue.pop(0)

            for neighborSimilarity, neighborNode in currentNode.friendList:
                cosine_similarity = self.getCosine_similarity(
                    vector1st=neighborNode.embeddingVector, 
                    vector2nd=subjectiveNode.embeddingVector
                )

                if len(neighborNode.friendList) < self.maxSize:
                    neighborNode.friendList.append((cosine_similarity, subjectiveNode))
                    neighborNode.friendList.sort(key=lambda x: x[0], reverse=True)
                    return True

                farthestSimilarity, _ = neighborNode.friendList[-1]
                if cosine_similarity > farthestSimilarity:
                    neighborNode.friendList[-1] = (cosine_similarity, subjectiveNode)
                    neighborNode.friendList.sort(key=lambda x: x[0], reverse=True)
                    return True

            queue.extend(node for _, node in currentNode.friendList)

        return False
    
    
    def findLocalMinimum(self,entryPoint,generations=None):
        """

        """
        #Case1 Not last layer
        #Case2 Last layer
        #

    
            
    def createAllConnectedNodes(self, node):
        """
        Recursively creates all connected friend nodes.
        """
        # Base case: If the node has no friends, stop recursion
        if not node.friendNodeIds:
            return

        # Process each child node recursively
        for childNodeId in node.friendNodeIds:
            if childNodeId not in self.nodes:  # Avoid re-adding existing nodes
                # Create the child node
                childNode = self.createNode(id=childNodeId)
                # Add the child node to the nodes dictionary
                self.nodes[childNode.id] = childNode
                # Recursively process the child node
                self.createAllConnectedNodes(node=childNode)

    def createNodesUpToDepth(self,depth,node):
        """
        Recursively creates friend nodes up to a specified depth.
        """
        # Base case: Stop recursion when depth reaches 0
        if depth <= 0:
            return

        # Process each child node recursively
        for childNodeId in node.friendNodeIds:
            if childNodeId not in self.nodes:  # Avoid re-adding existing nodes
                # Create the child node
                childNode = self.createNode(id=childNodeId)
                # Recursively process the child node with reduced depth
                self.createNodesUpToDepth(depth=depth - 1, node=childNode)

    def addToNN(subjectiveNode,objectiveNode,friendListSize=30):
        """Add objetive node to subjectiveNode's friend list"""
        #Case:have space to add
        #Case:No space to add
            #Case: more similar than farther Node
            #Case: less similar than farther Node
                #By BFS, I will try to find the node to add

        return
    
    def findNNStatically(self, entryPoint: Node, inputNode: Node):
        """ Find the local minimum from full graph based on cosine similarity """
        
        node = entryPoint
        nodeQueue = deque([node])  # Using deque for an efficient queue implementation
        isProceeding = True  # Flag to track if we should continue searching
        maxSimilarity = float('-inf')  # Initialize max similarity to a very low value
        similarestNode = None  # Initialize the most similar node variable

        while nodeQueue:  # While there are nodes to process in the queue
            node = nodeQueue.pop()  # Get the next node to process

            # Calculate similarity with the current node
            parentSimilarity = self.getCosine_similarity(node.vector, inputNode.vector)

            # Base case: if no children, it's a local minimum
            if len(node.friendList) == 0:
                self.addToNN(subjectiveNode=node, objectiveNode=inputNode)
                break

            else:
                # Iterate over child nodes and calculate similarity
                for childNode in node.friendList:
                    childSimilarity = self.getCosine_similarity(childNode.vector, inputNode.vector)
                    if childSimilarity > parentSimilarity:  # Continue if child is more similar
                        isProceeding = True
                    
                    if childSimilarity > maxSimilarity:  # Update max similarity and most similar node
                        maxSimilarity = childSimilarity
                        similarestNode = childNode

                if similarestNode:
                    nodeQueue.append(similarestNode)  # Add the most similar child to the queue

                # If no child is more similar than the parent, it's a local minimum
                if not isProceeding:
                    self.addToNN(subjectiveNode=node, objectiveNode=inputNode)
                    break  # Exit the loop since we've found the local minimum          
         
    def findNNDynamically(self, entryPoint, inputNode, depth):
        """
        Finds the local minimum in a Navigable Small World graph
        by shifting subgraph (disk-based).

        Parameters:
        - entryPoint: The starting node ID for the search.
        - inputNode: The target node for similarity comparison.
        - depth: The depth to which the subgraph should be expanded.

        Returns:
        - node: The node representing the local minimum.
        """
        # Create the starting node from the entry point
        parentsNode = self.createNode(id=entryPoint)           
        nodeQueue = [parentsNode]  # Initialize the nodeQueue with the entry node

        nearestSimilarity = float('-inf')  # Initialize with the worst similarity
        nnId = None  # To store the ID of the best matching node
        
        while nodeQueue:     
            node = nodeQueue.pop(0)  # BFS: Get the next node from the queue
            # Get similarity score between the current node and the inputNode
            similarityWithParents = self.getCosine_similarity(vector1st=node.embeddingVector, vector2nd=inputNode.embeddingVector)

            # Initialize isLocalMinimum to track if a better node is found
            isLocalMinimum = True

            # Compare all child nodes to get the best similarity
            for childNodeId in node.friendList:

                # If child node is not in memory, load its subgraph
                if childNodeId not in self.nodes:                    
                    self.createSubGraph(entryPoint=node.id, depth=depth)

                childNode = self.nodes[childNodeId]
                similarityWithChild = self.getCosine_similarity(vector1st=childNode.embeddingVector, vector2nd=inputNode.embeddingVector)

                # If the child node is closer than the parent node, update
                if similarityWithChild > similarityWithParents:
                    isLocalMinimum = False  # A better node has been found
                
                # If a new nearest neighbor is found, update the nearest similarity and node ID
                if similarityWithChild > nearestSimilarity:
                    nearestSimilarity = similarityWithChild
                    nnId = childNodeId  # Mark the best matching neighbor

            # If the parent node remains the closest, add the inputNode to its neighbors
            if isLocalMinimum:
                self.addToNN(subjectiveNode=node, objectiveNode=inputNode)
                return node  # Return the local minimum (parent node)

            # If we reach a leaf node or no more children, stop and return the current node  
            if len(node.friendList) == 0:
                self.addToNN(subjectiveNode=node, objectiveNode=inputNode)
                return node  # Return the leaf node as the local minimum

            # Update the nodeQueue with the best matching child node (if any)
            if nnId:
                bestNode = self.nodes[nnId]
                nodeQueue.append(bestNode)  # Add the best node to the queue for further exploration

        # If local minimum not found after traversing the graph, return an error message
        raise ValueError("Local minimum not found.")



    

    def createNode(self,id)->Node:
        return Node





    def dynamicallyGetNodes(self,entryPoint:id,generations=None:id):
        """
        Dynamically get node via id
        Params:
            entryPoint:
            generations:
        Returns:
        """
        #Case1: No generations(Not last layer)
            #until
        #Case2: With param of generations(Last layer)            





        







    
    def appendingToFriendList(self,nodeOne,nodeTwo,similarity):
        """
        If similarity is more than threshold,attempting adding edge
        
        Arguments:
            nodeOne: first candidate node 
            nodeTwo: secound candidate node 
        
        Returns:
            float: The cosine similarity between the two vectors.
        """
        #if similary than farthest node in friendList,then replace
        if nodeOne.friendNode[-1][0] < similarity:
            nodeOne.friendNode[-1] = (similarity,nodeTwo)
            nodeOne.friendNode
            HeapSorter.heapsort(nodeOne.friendNode)


    def getCosine_similarity(self,vector1st:np.ndarray, vector2nd:np.ndarray):
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