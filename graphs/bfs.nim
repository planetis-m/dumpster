import std / [deques, with]

type
  Edge = object
    neighbor {.cursor.}: Node

  Node {.acyclic.} = ref object
    neighbors: seq[Edge]
    label: string
    visited: bool

  Graph = object
    nodes: seq[Node]

proc addNode(self: var Graph; label: string): Node =
  self.nodes.add(Node(label: label))
  result = self.nodes[^1]

proc addEdge(self: Graph; source, neighbor: Node) =
  source.neighbors.add(Edge(neighbor: neighbor))

proc breadthFirstSearch(graph: Graph; source: Node): seq[string] =
  var queue: Deque[Node]
  queue.addLast(source)

  result = @[source.label]
  source.visited = true

  while queue.len > 0:
    let node = queue.popFirst()
    for edge in node.neighbors:
      let neighborNode = edge.neighbor
      if not neighborNode.visited:
        queue.addLast(neighborNode)
        neighborNode.visited = true
        result.add(neighborNode.label)

proc main =
  var graph: Graph

  let nodeA = graph.addNode("a")
  let nodeB = graph.addNode("b")
  let nodeC = graph.addNode("c")
  let nodeD = graph.addNode("d")
  let nodeE = graph.addNode("e")
  let nodeF = graph.addNode("f")
  let nodeG = graph.addNode("g")
  let nodeH = graph.addNode("h")

  with(graph):
    addEdge(nodeA, neighbor = nodeB)
    addEdge(nodeA, neighbor = nodeC)
    addEdge(nodeB, neighbor = nodeD)
    addEdge(nodeB, neighbor = nodeE)
    addEdge(nodeC, neighbor = nodeF)
    addEdge(nodeC, neighbor = nodeG)
    addEdge(nodeE, neighbor = nodeH)
    addEdge(nodeE, neighbor = nodeF)
    addEdge(nodeF, neighbor = nodeG)

  let nodesExplored = breadthFirstSearch(graph, source = nodeA)
  echo(nodesExplored)

main()
