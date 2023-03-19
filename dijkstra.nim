##[
  Exploring  Dijkstra,
  The data example is from
  https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
  by CCS, converted to nim by planetis-m
  Dijkstra's single source shortest path algorithm.
  The program uses an adjacency matrix representation of a graph
  This Dijkstra algorithm uses a priority queue to save
  the shortest paths. The queue structure has a data
  which is the number of the node,
  and the priority field which is the shortest distance.
  PS: all the pre-requisites of Dijkstra are considered
  $ nim c -r file_name.nim
  $ ./an_executable.EXE
  Code based from : Data Structures and Algorithms Made Easy: Data Structures and Algorithmic Puzzles, Fifth Edition (English Edition)
  pseudo code written in C
  This idea is quite different: it uses a priority queue to store the current
  shortest path evaluted
  The priority queue structure built using a list to simulate
  the queue. A heap is not used in this case.
]##
import std/[enumerate, heapqueue, sequtils]

type
  Node = tuple
    data: int
    priority: int

proc `==`(a, b: Node): bool = a.data == b.data
proc `<`(a, b: Node): bool = a.priority < b.priority

proc all_adjacents[T](g: seq[seq[T]], v: int): seq[int] =
  # give a NODE v, return a list with all adjacents
  # Take care, only positive EDGES
  var temp: seq[int] = @[]
  for i in 0 ..< g.len:
    if g[v][i] > 0:
      temp.add(i)
  return temp

proc print_solution[T](dist: seq[T]) =
  echo "Vertex \tDistance from Source"
  for node in 0 .. (dist.len - 1):
    echo "\n ", node, " ==> \t ", dist[node]

proc print_paths_dist[T](path: seq[T], dist: seq[T]) =
  # print all  paths and their cost or weight
  echo "\n Read the nodes from right to left (a path): \n"
  for node in 1 ..< path.len:
    echo "\n ", node, " "
    var i = node
    while path[i] != -1:
      echo " <= ", path[i], " "
      i = path[i]
    echo "\t PATH COST: ", dist[node]

proc updating_priority[T](prior_queue: var HeapQueue[T], item: T) =
  # Change the priority of a value/node ... exist a value, change its priority
  let pos = prior_queue.find(item)
  prior_queue.del(pos)
  prior_queue.push item

proc dijkstra(g: seq[seq[int]], s: int) =
  # check structure from: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
  # s: source for all nodes
  # Two results are obtained ... cost and paths
  var pq_queue: HeapQueue[Node] # creating a priority queue
  push(pq_queue, (data: s, priority: 0)) # goes s with priority 0
  var n = g.len

  var dist: seq[int] = newSeqWith[int](n, -1) # dist with -1 instead of INIFINITY
  var path: seq[int] = newSeqWith[int](n, -1) # previous node of each shortest paht

  # Distance of source vertex from itself is always 0
  dist[s] = 0

  while pq_queue.len > 0:
    var (v, _) = pop(pq_queue)
    # for all W adjcents vertices of v
    var adjs_of_v = all_adjacents(g, v) # all_ADJ of v ....
    # echo "\n ADJ ", v, " is ", adjs_of_v"
    var new_dist = 0
    for w in adjs_of_v.items:
      new_dist = dist[v] + g[v][w]
      if dist[w] == -1:
        dist[w] = new_dist
        push(pq_queue, (data: w, priority: dist[w]))
        path[w] = v # collecting the previous node -- lowest weight
      if dist[w] > new_dist:
        dist[w] = new_dist
        updating_priority(pq_queue, (data: w, priority: dist[w]))
        path[w] = v

  # print the constructed distance array
  print_solution(dist)
  # echo("\n \n Previous node of shortest path: ", path)
  print_paths_dist(path, dist)

# Solution Expected
# Vertex   Distance from Source
# 0                0
# 1                4
# 2                12
# 3                19
# 4                21
# 5                11
# 6                9
# 7                8
# 8                14

proc main =
  # adjacency matrix = cost or weight
  let graph_01 = @[
    @[0, 4, 0, 0, 0, 0, 0, 8, 0],
    @[4, 0, 8, 0, 0, 0, 0, 11, 0],
    @[0, 8, 0, 7, 0, 4, 0, 0, 2],
    @[0, 0, 7, 0, 9, 14, 0, 0, 0],
    @[0, 0, 0, 9, 0, 10, 0, 0, 0],
    @[0, 0, 4, 14, 10, 0, 2, 0, 0],
    @[0, 0, 0, 0, 0, 2, 0, 1, 6],
    @[8, 11, 0, 0, 0, 0, 1, 0, 7],
    @[0, 0, 2, 0, 0, 0, 6, 7, 0],
  ]
  let graph_02 = @[
    @[0, 2, 0, 6, 0],
    @[2, 0, 3, 8, 5],
    @[0, 3, 0, 0, 7],
    @[6, 8, 0, 0, 9],
    @[0, 5, 7, 9, 0],
  ]
  # data from https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
  # The graph:
  #       2    3
  #   (0)--(1)--(2)
  #   |    / \    |
  #  6|  8/   \5  |7
  #   |  /     \  |
  #   (3)-------(4)
  #        9

  # Let us create following weighted graph
  # From https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-gre
  # Let us create following weighted graph
  # From https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/?ref=lbp
  #                  10
  #             0--------1
  #             |  \     |
  #            6|   5\   |15
  #             |      \ |
  #             2--------3
  #                 4

  let graph_03 = @[
    @[0, 10, 6, 5],
    @[10, 0, 0, 15],
    @[6, 0, 0, 4],
    @[5, 15, 4, 0],
  ]
  # To find number of coluns
  # var cols = an_array[0].len
  for i, graph in enumerate(1, [graph_01, graph_02, graph_03]):
    # allways starting by node 0
    let start_node = 0
    echo "\n\n Graph ", i, " using Dijkstra algorithm (source node: ", start_node, ")"
    dijkstra(graph, start_node)

  echo "\n BYE -- OK"

main()
