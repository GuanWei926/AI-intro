import csv
from collections import defaultdict
import heapq

edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

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

def read_heuristics(heuristicFile):
    heuristic = defaultdict(dict)
    with open(heuristicFile, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            node = int(row['node'])
            dist_ID1 = float(row['1079387396'])
            dist_ID2 = float(row['1737223506'])
            dist_ID3 = float(row['8513026827'])
            heuristic[node]['1079387396'] = dist_ID1
            heuristic[node]['1737223506'] = dist_ID2
            heuristic[node]['8513026827'] = dist_ID3
    return heuristic

def astar(start, end):
    # Begin your code (Part 4)
    '''
    First of all, I utilize two custom functions. One of the functions is read_graph, to parse the edges.csv file 
    and organize the data into a dictionary format (each dictionary includes one list). The other is read_heuristics,
    to parse the heuristicFile.csv and organize the data into a dictionary format (each dictionary includes another
    dictionary). Subsequently, I use the list to create a priority queue so as to store the node I traverse. 
    Additionally, I establish a set named visited to track whether a node has been traversed. 
    Next, I implement A* search using a while loop. During each iteration, I use heappop to extract the node with the 
    highest priority from the priority queue, which maintains the priority queue's property, and add it to the visited
    set. If the node I currently traverse is end node, then I return the path, distance and number of visited nodes. 
    Otherwise, put the evaluated value (the sum of the movement cost from starting node to current node and the estimated 
    movement cost from current node to end node), updated distance, next unvisited nodes, and updated path into the 
    priority queue. The traversal continues until the priority queue becomes empty. If the end node is not reached, the
    function returns an empty list, -1, and the count of visited nodes.
    '''
    graph = read_graph(edgeFile)
    heuristic = read_heuristics(heuristicFile)

    priority_queue = [(heuristic[start][str(end)], 0, start, [start])]  # (sum of the actual cost (dist) and the heuristic estimate
                                                                        #    , total distance, current node, path)
    visited = set()

    while priority_queue:
        _, dist, node, path = heapq.heappop(priority_queue)
        visited.add(node)
        if node==end:
            num_visited = len(visited)-1
            return path, dist, num_visited
        
        for next, weight, _ in graph.get(node, []):
            if next not in visited:
                updated_dist = dist + weight
                heapq.heappush(priority_queue, (updated_dist + int(heuristic[next][str(end)]), updated_dist, next, path+[next]))

    # If we do not reach the end node
    num_visited = len(visited)-1
    return [], -1,  num_visited
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
