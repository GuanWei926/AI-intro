import csv
from collections import defaultdict
import heapq

edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

def read_graph(graphfile):
    graph = defaultdict(list)
    max_speed_limit = 0
    with open(graphfile, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start = int(row['start'])
            end = int(row['end'])
            distance = float(row['distance'])
            speed_limit = float(row['speed limit'])
            graph[start].append((end, distance, speed_limit))
            if speed_limit > max_speed_limit:
                max_speed_limit = speed_limit
    return graph, max_speed_limit

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

def astar_time(start, end):
    # Begin your code (Part 6)
    '''
    First of all, I utilize two custom functions. One of the functions is read_graph, to parse the edges.csv file 
    and organize the data into a dictionary format (each dictionary includes one list). The other is read_heuristics,
    to parse the heuristicFile.csv and organize the data into a dictionary format (each dictionary includes another
    dictionary). Subsequently, I use the list to create a priority queue so as to store the node I traverse. 
    Additionally, I establish a set named visited to track whether a node has been traversed. Note that we should perform
    unit conversion before using the limit of the speed.
    Next, I implement A*_time search using a while loop. During each iteration, I use heappop to extract the node with the 
    highest priority from the priority queue, which maintains the priority queue's property, and add it to the visited
    set. If the node I currently traverse is end node, then I return the path, time and number of visited nodes. 
    Otherwise, put the evaluated value (the sum of the movement time from starting node to current node and the estimated 
    movement cost divides the maximum speed limit from current node to end node), updated time, next unvisited nodes, and 
    updated path into the priority queue. The traversal continues until the priority queue becomes empty. If the end node
     is not reached, the function returns an empty list, -1, and the count of visited nodes.
    '''
    graph, max_speed_limit = read_graph(edgeFile)
    heuristic = read_heuristics(heuristicFile)
    max_speed_limit = max_speed_limit*1000/3600

    # (sum of the actual time and the heuristic estimate (time)
    #   , total time, current node, path)
    priority_queue = [(heuristic[start][str(end)]/max_speed_limit, 0, start, [start])]  
    visited = set()

    while priority_queue:
        _, time, node, path = heapq.heappop(priority_queue)
        if node in visited:
            continue
        
        visited.add(node)
        if node==end:
            num_visited = len(visited)-1
            return path, time, num_visited
        
        for next, weight, speed_limit in graph.get(node, []):
            if next not in visited:
                speed_limit = speed_limit * 1000 / 3600
                updated_time = time + weight / speed_limit
                heapq.heappush(priority_queue, (updated_time + float(heuristic[next][str(end)]) / max_speed_limit, updated_time, next, path+[next]))

    # If we do not reach the end node
    num_visited = len(visited)-1
    return [], -1,  num_visited
    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
