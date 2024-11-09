#set the number of edge to have 
#edge management
    #find closest Node -> count -> if there is space ,add-> 
import numpy as np
from ...nodeManager.sortManager.heap import Heap




#Adding link: 
    # traverse link 
        #adding isiting list
        #exchange with best
class NSW:
    """each element is """
    def __init__(self,maximumFriend:int) -> None:
        self.maximumFriend = maximumFriend
        self.Nodes = []

    #<Creating graph>-----------------------------------------
    def createGraph(self):



    #<finding node in graph>-----------------------------------
    def findClosestNodeInNSW(self,entryNodeId:int,queryVector:np.array,ef:int,moreThanThresholds:list,threshold:int=1,isLastLayer=False)->tuple:
        """
        find Nearest Node list in this layer
        Argument:
            -entryNodeId(int):id of entry node
            -queryVector
            -ef(efConstruction):factor to controll breadth of exploration
        Return:
            metaDataOfVector(tuple)(distance,id of Node)
        """
        
        #get distance
        nearestDistance = np.linalg.norm(self.Nodes[entryNodeId].embeddingVector - queryVector)
        entryData = (nearestDistance,entryNodeId)
        #Prelimianary best node(consisting distance and node)
        nns = [entryData]
        #get visited node set
        visit = set(entryData)
        # List of candidates
        candidates = [entryData]
        #sort candidate node by distance
        heapify(candidates)

        #while candidateNodes exist
        while candidates:
            #get closest Node and implement max sort
            currentBest  = heapop(candidates)
            #+++if 
            if nns[-1][0] > candidateNode[0]:
                break
            #loop through all nearest neighbors to the candidate vector
            candidateNode = currentBest[1]

            for friendNode in candidateNode.friendList:
                distance = np.linalg.norm(friendNode.embeddingVector-queryVector)
                
                if (distance,friendNode) not in visit:
                    visit.add((distance,friendNode))

                    # push only "better" vectors into candidate heap
                    if distance < nns[-1][0] or len(nns) < ef:
                        heappush(candidates,(distance,friendNode))
                        insort(nns,(distance,friendNode))
                        if len(nns) > ef:
                            nns.pop()
        return nns
    





    def addNode(self,node):
        self.Nodes.append(node)

    def popNodeByID(self,id):
        #traverse all of Node in Nodes and if found Node whose id is identical with argment yes delete it 

    




    #by initialize 