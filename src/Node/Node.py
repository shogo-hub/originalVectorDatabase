from ..sortManager.heapSort.heap import Heap


class Node:
    
    def __init__(self,id:int,friendNode,vector,next_layer_index:int):
        self.friendList = []   #[(similairty,id)]
        self.friendList = []   #[id:Node]
        self.id = id
        self.vector = None
        self.next_layer_index = next_layer_index
    
    
    def getId(self):
        return self.id
    
    def getVector(self):
        return self.vector
    
    def getfriendList(self):
        return self.friendList
    
    def getNext_layer_index(self):
        return self.next_layer_index

    def sortFriendList(self):
        
    