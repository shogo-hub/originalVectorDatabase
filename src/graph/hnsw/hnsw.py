
from nodeManager.nodeManager.nodeCreator import Node
# set the number of layer 
L= 5
#3-tuple containing the vector, 
#   a list of indexes the vector links to within the graph, 
#   and the index for the corresponding node in the layer below

class HNSW:
    """This is the code to create HNSW"""
    def __init__(self,L:int,efc:int=10):
        """
        Explanation:
            HNSW  is the list consist of NSW
            NSW is the 2 dimensional list(feature is Node and id at the graph)
        Argument:
            -L: The number of layer in HNSW


        """
        self.efc = efc
        self.L = L
        #order is layer (from top to bottom)
        self.hnsw = [[]for _ in range(L)]
    #<cerating algorithm>----------------------------------
    def creatingHNSW(self):
        #later think what to do (all of vector and set the number of layer or so ?)
    
    def insertingNode(self,vector,referenceToSource:int,efc=10):
        """
        """
        #case1:if hnsw is empty, insert the node to all layer
        if not self.hnsw[0]:
            idAtThisGraph = None
            for nsw in self.hnsw[::-1]:
                graphElement = (idAtThisGraph,node)
                nsw.append(graphElement)
            return 
        
        #case2:
        #decide the layer to input
        l = ""
        for n , nsw in enumerate(self.hnsw):
            #if  layer is higher than the layer to start
            if n <l:
                #get distance with entry point
                #find the closest node to go to entry point to next layer
                
            #add node to rest of node
            else:
                #set node 
                次の階層への接続方法の詳細はのちに考える
                node = Node(id=referenceToSource,embeddingVector=vector,next_layer_index=(len(self.hnsw[n+1]) if n < self._L-1 else None))





    
    
    
    
    #<searching algorithm>-------------------------------------------  
    def searchNearestNode(self,queryVector,ef=1):
        """
        Find nearest node out of HNSW
        """
        #if there is no graph ,return none
        if not self.hnsw[0]:
            return []
        #set best node to the entry point 
        bestNode = 0

        #iterate throush each layer(NSW) in HNSW
        for nsw in self.hnsw:
            bestDistance ,bestNode  = nsw.findClosestNodeInNSW(bestNode,queryVector,ef)
            #get the id of node one below
            if nsw[bestNode.id].next_layer_index:
                bestNode = nsw[bestNode.id].next_layer_index
            
            else:
                nsw.findClosestNodeInNSW(bestNode,queryVector,ef)

    def searchLayer(self,queryVector,entry,ef=1):
        """
        """
        


