"""
Final Project: Al-GO-rithms Delivery Challenge
Author: Rayner Sarmiento
"""
import heapq

from collections import defaultdict

def algorithm_1(graph:dict, start:str, end:str):
    """Finds the shortest path between start and end using Dijkstra's Algorithm."""

    #Initialize distances with infinity for all nodes except start
    distances = {node: float('inf') for node in graph}
    distances[start] = 0 
    #Tracking predecessors to reconstruct the path 
    predecessors = {node: None for node in graph}
    #priority queue to store distance, node pairs 
    priority_queue = [(0, start)]
    #set to track visited nodes 
    visited = set()

    while priority_queue: 
        #Extract node with minimum distance 
        current_distance, current_node = heapq.heappop(priority_queue)
        #skip if already visited 
        if current_node in visited: 
            continue 
        visited.add(current_node)
        #stop if reached end node
        if current_node== end: 
            #reconstruct path 
            path = []
            current = end 
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            return path, distances[end]
        
        #explore neighbors 
        for neighbor, weight in graph[current_node]: 
            if neighbor not in visited: 
                #calculate distance to neighbor via current node 
                new_distance = current_distance + weight 
                #Update if a shorter path is found 
                if new_distance < distances[neighbor]: 
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node 
                    heapq.heappush(priority_queue, (new_distance, neighbor))
    #If no Path exists to the end node 
    return [],float('inf')
    
#Algorithm 2 
def algorithm_2(graph: dict, hub:str): 
    """Computes the Minimum Spanning Tree(MST)"""
    #Check if hub exits in the graph 
    if hub not in graph: 
        return [], float('inf')
    
    #initialize MST and priority queue 
    mst = []
    priority_queue = [(0,None,hub)] #(weight, parent, node)
    visited = set()
    total_cost = 0 

    while priority_queue: 
        #Extract edge with minimum weight 
        weight, parent, current_node = heapq.heappop(priority_queue)
        #Skip if already visited 
        if current_node in visited:
            continue
        visited.add(current_node)
        # Add edge to MST skip for the hub as it has no parent
        if parent is not None: 
            mst.append((parent, current_node, weight))
            total_cost += weight

            #explore neighbors
        for neighbor, weight in graph[current_node]:
            if neighbor not in visited:
                heapq.heappush(priority_queue, (weight, current_node, neighbor))

    # Check if graph is fully connected
    if len(visited) != len(graph):
        return [], float('inf')  # Graph is disconnected

    return mst, total_cost


#Algorithm 3 

def algorithm_3(graph: dict, hub:str, edges_to_remove:list, edges_to_add:list):
    """Updates the graph by removing and adding edges, then computes the MST using prim's Algorithm
    Args: 
    graph: dictionary 
    hub: string 
    edges_to_remove: list of strings
    edges_to_add: list of tuples
    """
    #Create a copy of the graph 
    updated_graph = defaultdict(list)
    for node in graph: 
        for neighbor, weight in graph[node]: 
            updated_graph[node].append((neighbor, weight))
            updated_graph[neighbor] #ensure all nodes exist in the graph
    
    #Process for edges removal
    for edge in edges_to_remove: 
        node_1, node_2 = edge.split('-')
        if node_1 in updated_graph and node_2 in updated_graph: 
            updated_graph[node_1] = [(n, w) for n, w in updated_graph[node_1] if n != node_2]
            updated_graph[node_2] = [(n, w) for n, w in updated_graph[node_2] if n != node_1]   

    #Process for edges additions 
    for node_1, node_2, weight in edges_to_add: 
        updated_graph[node_1].append((node_2, weight))
        updated_graph[node_2].append((node_1, weight))
    #convert to dict for compatibility with algorithm_2 
    updated_graph = dict(updated_graph)
    #Compute MST on updated graph
    mst, total_cost = algorithm_2(updated_graph, hub)
    # Check if graph is still connected
    if len(mst) < len(updated_graph) - 1:
        return [], float('inf')  # Graph is disconnected

    return mst, total_cost 


# testing
if __name__ == "__main__":

    graph = {
        "A": [("B", 4), ("C", 2)],
        "B": [("A", 4), ("C", 1), ("D", 5)],
        "C": [("A", 2), ("B", 1), ("D", 8), ("E", 10)],
        "D": [("B", 5), ("C", 8), ("E", 2)],
        "E": [("C", 10), ("D", 2)]
    }

    # Test Algorithm 1
    print("=== Testing Algorithms fuctionalities ")

    path, cost = algorithm_1( graph, "A", "E")
    print(f"Algorithm 1 (A -> E): Path: {path}, Cost: {cost}")
    path, cost = algorithm_1( graph, "A", "B")
    print(f"Algorithm 1 (A -> B): Path: {path}, Cost: {cost}")

    # Test Algorithm 2
    mst, cost = algorithm_2( graph, "A")
    print(f"Algorithm 2 (Hub A): MST: {mst}, Cost: {cost}")

    # Test Algorithm 3
    mst, cost = algorithm_3( graph, "A", ["C-E"], [("B", "E", 3)])
    print(f"Algorithm 3 (Hub A, remove C-E, add B-E=3): MST: {mst}, Cost: {cost}")


    #Test Case from the part-1 

            # Test cases para Algorithm 1: Lowest Cost Delivery
    print("=== Testing Algorithm 1: Lowest Cost Delivery ===")

    # Test Case 1: Standard Graph
    graph_1 = {
        "A": [("B", 4), ("C", 2)],
        "B": [("A", 4), ("C", 1), ("D", 5)],
        "C": [("A", 2), ("B", 1), ("D", 8), ("E", 10)],
        "D": [("B", 5), ("C", 8), ("E", 2)],
        "E": [("C", 10), ("D", 2)]
    }
    path, cost = algorithm_1(graph_1, "A", "E")
    print(f"Test 1: Standard Graph (A -> E): Path: {path}, Cost: {cost}")

    # Test Case 2: Minimal Graph
    graph_2 = {
        "X": [("Y", 3)],
        "Y": [("X", 3)]
    }
    path, cost = algorithm_1(graph_2, "X", "Y")
    print(f"Test 2: Minimal Graph (X -> Y): Path: {path}, Cost: {cost}")

    # Test Case 3: No Path Exists
    graph_3 = {
        "A": [("B", 1)],
        "B": [("A", 1)],
        "C": []
    }
    path, cost = algorithm_1(graph_3, "A", "C")
    print(f"Test 3: No Path Exists (A -> C): Path: {path}, Cost: {cost}")

    # Test Case 4: Multiple Equal-Cost Paths
    graph_4 = {
        "A": [("B", 2), ("C", 2)],
        "B": [("A", 2), ("D", 2)],
        "C": [("A", 2), ("D", 2)],
        "D": [("B", 2), ("C", 2)]
    }
    path, cost = algorithm_1(graph_4, "A", "D")
    print(f"Test 4: Multiple Equal-Cost Paths (A -> D): Path: {path}, Cost: {cost}")