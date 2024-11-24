from ..heapSort.heap import HeapSorter

class PriorityQueue:
    def __init__(self,maxSize):
        """This is the descending order of priority queue of static array"""
        self.maxSize = maxSize
        self.heap = [] * self.maxSize

    
    def getSize(self):
        return len(self.heap)
    

    def sortArray(self):
        HeapSorter.heapsort(arr=self.heap)

    def insert(self, value:set):
        if self.heap[-1] is not None:
            raise Exception("Priority Queue is full.")
        self.heap.append(value)
        HeapSorter.heapsort(arr=self.heap)
        
      # Optional: You could also use heapify after each insert to keep the queue in order.
    
    def repalceTail(self,newTuple:tuple):
        self.pop_tail()
        self.insert(value=newTuple)
    
    
    def pop(self):
        if len(self.heap) == 0:
            raise Exception("Priority Queue is empty.")
        root = self.heap[0]
        # Swap the root with the last element, then heapify
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        self.heap.pop()
        HeapSorter.maxHeapify(self.heap, len(self.heap), 0)  # Ensure the heap property is maintained
        return root
    

    def peek(self):
        if len(self.heap) == 0:
            raise Exception("Priority Queue is empty.")
        return self.heap[0]  # Root element of the max-heap or min-heap
    

    def tail(self):
        if len(self.heap) == 0:
            raise Exception("Priority Queue is empty.")
        return self.heap[-1]  # Root element of the max-heap or min-heap

    def size(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0

    def pop_tail(self):
            """Remove the last element (tail) of the queue."""
            if len(self.heap) == 0:
                raise Exception("Priority Queue is empty.")
            tail = self.heap.pop()  # Remove last element
            return tail
    





    