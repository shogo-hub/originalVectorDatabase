from ..sortManager.heapSort.heap import Heap


class Node:
    
    def __init__(self,id:int,friendNode,embeddingVector,next_layer_index:int):
        self.friendList = []   # later replace list with heap
        self.idToRefference = id
        self.embeddingVector = embeddingVector
        self.next_layer_index = next_layer_index
    

    def getId(self):
        return self.id
    
    def getVector(self):
        return self.
    
    def getfriendList(self):
        return self.friendList
    
    def getNext_layer_index(self):
        return self.next_layer_index
    
    