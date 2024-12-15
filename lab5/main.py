from collections import defaultdict, deque


def solve(V, A, L):
    V = topological_sort(V, A)
    print(f"Топологическая сортировка: {V}")
    OPT = {V[0]: 0}; x = {V[0]: V[0]}

    for i, v in enumerate(V):
        if i == 0:
            continue
        
        options = dict()
        for a, l in zip(A, L):
            if a[1] == v:
                options[a[0]] = OPT[a[0]] + l
                
        max_key, max_value = max(options.items(), 
                                 key=lambda x: x[1])
        OPT[v] = max_value; x[v] = max_key   
        
    length = OPT[V[-1]]; way = [V[-1]]
    while way[0] != V[0]:
        way.insert(0, x[way[0]])
    
    return length, way
    

def topological_sort(vertices, edges):
    graph = defaultdict(list)
    indegree = {v: 0 for v in vertices}
   
    for start, end in edges:
        graph[start].append(end)
        indegree[end] += 1
 
    queue = deque([v for v in vertices if indegree[v] == 0])
    topological_order = []
    
    # Алгоритм Кана
    while queue:
        node = queue.popleft()
        topological_order.append(node)
        
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
          
            if indegree[neighbor] == 0:
                queue.append(neighbor)
  
    if len(topological_order) == len(vertices):
        return topological_order
    else:
        return "Граф содержит цикл и не может быть топологически отсортирован."


if __name__ == "__main__":
    V = ['s', 'a', 'b', 'c', 'd', 't']
    A = [
        ('s', 'a'), ('s', 'c'), 
        ('a', 'b'), ('b', 'd'), 
        ('b', 't'), ('c', 'a'), 
        ('c', 'd'), ('d', 't')
    ]
    L = [3, 2, 4, 1, 2, 2, 2, 1]
    
    result = solve(V, A, L)
    print("Result:", result)