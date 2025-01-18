import math

class HeapSorter:
    """Implement heapsort for static array in descending order"""
    
    @staticmethod
    def left(i):
        return 2 * i + 1
    
    @staticmethod
    def right(i: int) -> int:
        return 2 * i + 2
    
    @staticmethod
    def parent(i: int) -> int:
        return math.floor((i - 1) / 2)
    
    @staticmethod
    def maxHeapify(arr, heapEnd, i: int) -> None:
        """
        Implement max heapify for tuple
        """
        biggest = i
        # get index of left
        l = HeapSorter.left(i)
        # get index of right
        r = HeapSorter.right(i)
        
        # compare for max-heap
        if l < heapEnd and arr[l][0] > arr[biggest][0]: 
            biggest = l
        if r < heapEnd and arr[r][0] > arr[biggest][0]: 
            biggest = r

        if biggest != i:
            arr[i], arr[biggest] = arr[biggest], arr[i]  # Swap
            HeapSorter.maxHeapify(arr, heapEnd, biggest)

    @staticmethod
    def buildMaxHeap(arr):
        """
        Build max heap for HNSW
        """
        # Get middle (since elements under middle are all leaf nodes)
        middle = HeapSorter.parent(len(arr))

        for i in range(middle, -1, -1):
            HeapSorter.maxHeapify(arr, len(arr), i)

    @staticmethod
    def heapsort(arr):
        """
        """
        # Build max heap
        HeapSorter.buildMaxHeap(arr)

        heapEnd = len(arr) - 1

        while heapEnd > 0:
            arr[heapEnd], arr[0] = arr[0], arr[heapEnd]  # Swap
            heapEnd -= 1
            HeapSorter.maxHeapify(arr, heapEnd + 1, 0)

