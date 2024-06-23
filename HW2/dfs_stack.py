import csv
from collections import defaultdict

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

def dfs(start, end):
    # Begin your code (Part 2)
    '''
    First of all, I utilize a custom function, read_graph, to parse the edges.csv file and organize the data into a 
    dictionary format. Subsequently, I use the list to create a stack so as to store the node I traverse. 
    Additionally, I establish a set named visited to track whether a node has been traversed. 
    Next, I implement depth-first traversal using a while loop. During each iteration, I pop the top of stack to get 
    the current node, and add it to the visited set. If the node I currently traverse is end node, then I return the 
    path, distance and number of visited nodes. Otherwise, put the next unvisited nodes, updated path and updated 
    distance into the stack. The traversal continues until the stack becomes empty. If the end node is not reached, 
    the function returns an empty list, -1, and the count of visited nodes.
    '''
    graph = read_graph(edgeFile)

    stack = [(start, [start], 0)] #stores tuples of (node, path, distance)
    visited = set()
    while stack:
        node, path, dist = stack.pop()
        visited.add(node)
        if node==end:
            num_visited = len(visited)-1
            return path, dist, num_visited
        for next, weight, _  in graph.get(node, []):
            if next not in visited:
                stack.append((next, path + [next], dist+weight))
    
    # If we do not reach the end node
    num_visited = len(visited)-1
    return [], -1,  num_visited
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
