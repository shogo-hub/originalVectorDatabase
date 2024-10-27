import numpy as np
import random 


# L = 5 this is the number of layer 


#Hyper parameter :M(the amount of edge for each node to have),threshold (the threshold for similarity search)
#m_l :to controll how oftern to bring node to higher level 
#m_l:level mutiplier 


class Node:
    """This is the node class"""
    def __init__(self,id,embeddedVector) -> None:
        self.id = id 
        self.embeddedVector = embeddedVector


    def getId(self):
        return self.id

class Hnsw:
    """This is the algorithm to find node by similarity """


    def __init__(self) -> None:
        self.HNSW = {}
        self.cache = []

    #<graph construction>---------------
    def constructGraph(self,layers: int, Mmax: int, n: int, efConstruction: int, mL: float)->list:
        """
        Args:
            layers:hierarchical layers
            Mmax: the maximum edge for node can have
            n:the number of node to input
            efConstruction:
            mL:normalization factor for the level generation
        """
        #initialize HNSW
        self.HNSW = {
            "entrance":0,
            "layers":[Graph(), Graph(), Graph()]
        }
        #insert node to hnsw
        for _ in range(n):
            #node will be vector 
            node = []
            self.insert(HNSW,node,Mmax,efConstruction,mL)


        
        


    def chooseLevel(self):
        """
        This function take charge of choosing layer
        """
        #iterate through computing probability function and comparet it with uniform funcion 
            #if probabilty function> uniform function 
                #assing this layer and break

    def computeProbabilityFunction(self, m_L, level):
        """
        This function computes the probability that a node is connected to other nodes at a specific level 
        in a Hierarchical Navigable Small World (HNSW) graph.
            Example:
                If `m_L = 1.5` and `level = 2`, the probability of a node being connected at level 2 would be:

                Probability = exp(-2 / 1.5) * (1 - exp(-1 / 1.5))
                        = 0.5134 * 0.4865
                        = 0.2497 (approximately)
        Args:
            m_L: A parameter controlling how quickly the probability of connections between nodes decreases 
                as the level increases. It affects how many nodes will be connected at higher levels.
            level: The current level for which the connection probability is being calculated. 
                Higher levels generally mean fewer connections.

        Returns:
            A floating-point number representing the probability that a node will be connected 
            at the given level. This probability decreases as the level increases.
        """
        return np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))

    def computeUniformFucntion(self):
        """
        This function take charge of getting output of uniform function 
        """

    def manageFriendList(self):
        """
        This function manage friend list for new input at this layer
        """



    #decide which layer to be assingned 
    #1.use the function to compute , then go next layer
    #making edge
    def setDefaultProbability(self, M: int, m_L: float):
        """
        Sets default probabilities and cumulative neighbor counts for constructing
        a Hierarchical Navigable Small World (HNSW) graph.

        Args:
            M (int): Maximum number of neighbors (edges) that each node can have at
                    each level of the graph.
            m_L (float): A parameter that controls the probability of connections 
                        between levels. It influences how quickly the probability 
                        of node assignments decreases with increasing levels.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing two lists:
                - assignProbas (List[float]): A list of probabilities for assigning 
                a node to each hierarchical level. These probabilities decrease 
                exponentially with each level.
                - cumulativeNeighborPerLevel (List[int]): A list of cumulative 
                neighbor counts, where each entry corresponds to the total number 
                of edges a node can have at each level. The first level can 
                have up to M * 2 neighbors, while higher levels can have up to M 
                neighbors.
        
        The method iteratively calculates the probability of assigning a node 
        to each level until the probability falls below a threshold (1e-9), 
        indicating that further levels are negligible. The method ensures that 
        the first level has double the neighbors compared to the higher levels, 
        establishing a foundational layer for the HNSW graph structure.
        """
        # Set default value
        nn = 0
        cumulativeNeighborPerLevel = []
        level = 0
        assignProbas = []

        while True:
            # Calculate probability for the current level
            proba = np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))
            # If probability reaches a certain score, no probability to create a new level
            if proba < 1e-9:
                break
            else:
                assignProbas.append(proba)
            
            # Neighbor count: M at each layer but first layer has M*2
            if level == 0:
                nn += M * 2  # For the first level, add M * 2 neighbors
            else:
                nn += M 
            
            cumulativeNeighborPerLevel.append(nn)
            level += 1
            
        return assignProbas, cumulativeNeighborPerLevel


    def randomLevel(assignProbas:list,rng):
        #get random float from 

    #<find searching node>---------------------------
    #loop (find local minimum+go to next layer)
    #  



    def chooseLevel(self,probabilityFunction,vector):
        """
        Choose the level to assign vector to it
        """

        level = 0
        while random.uniform(0, 1) < self.level_multiplier and level < self.L - 1:
            level += 1
        return level

    def getEuclideanDistance(self,vec1,vec2):
        """
        Compute the Euclidian distance between two vectors]
        """

        return np.linalg.norm(np.array(vec1) - np.array(vec2))
    



