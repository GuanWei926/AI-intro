import csv
from collections import defaultdict
import heapq

edgeFile = 'edges.csv'

def read_graph(graphfile):
    graph = defaultdict(list)
    with open(graphfile, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start = int(row['start'])
            end = int(row['end'])
            distance = float(row['distance'])
            speed_limit = float(row['speed limit'])
            graph[start].append((end, distance, speed_limit))
    return graph

def ucs(start, end):
    # Begin your code (Part 3)
    '''
    First of all, I utilize a custom function, read_graph, to parse the edges.csv file and organize the data into a 
    dictionary format. Subsequently, I use the list to create a priority queue so as to store the node I traverse. 
    Additionally, I establish a set named visited to track whether a node has been traversed. 
    Next, I implement uniform cost search using a while loop. During each iteration, I use heappop to extract the
    node with the highest priority from the priority queue, which maintains the priority queue's property.Then, add 
    the node to the visited set. If the node I currently traverse is end node, then I return the path, distance and 
    number of visited nodes. Otherwise, I use heappush to enqueue the next unvisited nodes, along with updated path 
    and distance information, into the priority queue. The traversal continues until the priority queue becomes empty. 
    If the end node is not reached, the function returns an empty list, -1, and the count of visited nodes.
    '''
    graph = read_graph(edgeFile)

    priority_queue = [(0, start, [start])]  # Queue stores tuples of (distance, node, path)
    visited = set()

    while priority_queue:
        dist, node, path = heapq.heappop(priority_queue)
        visited.add(node)
        if(node == end):
            num_visited = len(visited)-1
            return path, dist, num_visited
        
        for next, weight, _  in graph.get(node, []):
            if next not in visited:
                heapq.heappush(priority_queue, (dist + weight, next, path + [next]))
    
    # If we do not reach the end node
    num_visited = len(visited)-1
    return [], -1,  num_visited
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
