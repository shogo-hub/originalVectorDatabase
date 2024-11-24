#set the number of edge to have 
#edge management
    #find closest Node -> count -> if there is space ,add-> 
from ...nodeManager.nodeManager.nodeCreator import Node
import numpy as np
from ...nodeManager.sortManager.priorityQueue.priorityQueue import PriorityQueue
from ...nodeManager.sortManager.heapSort.heap import HeapSorter




#Adding link: 
    # traverse link 
        #adding isiting list
        #exchange with best
class NSW:
    """each element is """
    def __init__(self,maximumFriend:int,maxSize:int) -> None:
        self.maximumFriend = maximumFriend
        self.nodes = {}    #{id:Node}
        self.priorityQueue = PriorityQueue(maxSize)

    #<Creating graph>-----------------------------------------
    def createNewGraph(self):
        """
        Take charge of creating new graph
        """



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
        #<<<Case1:No predecessor>>>
        if len(self.nodes) == None:
            self.nodes[entryNode.getId()] = entryNode
            return True
        

        #<<<Case2:With Predecessor>>>
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

    #<<Searching node from NSW>>
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


    def reconstructionConnectivity(self,friendListOfDeletingNode,threshold):
        """
        Rebuild the friend lists of all nodes that are affected by the deletion.
        
        Arguments:
            deletingNode: The node that was deleted and whose removal affects the friend lists of other nodes.
            
        Returns:
            None
        """

        length = len(friendListOfDeletingNode)

        for i in range(length):
            for j in range(i+1,length):
                vector1st = friendListOfDeletingNode[i].embeddingVector
                vector2nd = friendListOfDeletingNode[j].embeddingVector
                similarity = self.getCosine_similarity(vector1st=vector1st,vector2nd=vector2nd)
                if similarity>= threshold:
                    #add vector to friend List 
                    self.appendingToFriendList(parentNode=friendListOfDeletingNode[i],friendNode=friendListOfDeletingNode[j],similarity=similarity) #vector1st.friendList.append(vector2nd)
                    vector2nd.friendList.append(vector1st)

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