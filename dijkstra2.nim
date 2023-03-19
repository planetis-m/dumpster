# https://www.baeldung.com/java-dijkstra

import std/[sets, lists, tables, hashes, assertions]

type
  Graph* = ref object
    nodes: HashSet[Node]
  Node* = ref object
    name: string
    shortestPath: DoublyLinkedList[Node]
    distance: int = int.high
    adjacentNodes: Table[Node, int]

proc hash*(n: Node): Hash =
  hash(n.name)

proc addNode*(g: Graph, nodeA: Node) =
  g.nodes.incl(nodeA)

proc addDestination*(nodeA: Node, nodeB: Node, distance: int) =
  nodeA.adjacentNodes[nodeB] = distance

proc calculateMinimumDistance*(evaluationNode: Node, edgeWeigh: int, sourceNode: Node) =
  var sourceDistance = sourceNode.distance
  if sourceDistance + edgeWeigh < evaluationNode.distance:
    evaluationNode.distance = sourceDistance + edgeWeigh
    var shortestPath = sourceNode.shortestPath.copy
    shortestPath.add(sourceNode)
    evaluationNode.shortestPath = shortestPath

proc getLowestDistanceNode*(unsettledNodes: HashSet[Node]): Node =
  var lowestDistanceNode: Node
  var lowestDistance = int.high
  for node in unsettledNodes:
    var nodeDistance = node.distance
    if nodeDistance < lowestDistance:
      lowestDistance = nodeDistance
      lowestDistanceNode = node
  result = lowestDistanceNode

proc calculateShortestPathFromSource*(graph: Graph, source: Node): Graph =
  source.distance = 0
  var settledNodes = initHashSet[Node]()
  var unsettledNodes = initHashSet[Node]()
  unsettledNodes.incl(source)
  while unsettledNodes.len > 0:
    var currentNode = getLowestDistanceNode(unsettledNodes)
    unsettledNodes.excl(currentNode)
    for (adjacentNode, edgeWeigh) in currentNode.adjacentNodes.pairs:
      if adjacentNode notin settledNodes:
        calculateMinimumDistance(adjacentNode, edgeWeigh, currentNode)
        unsettledNodes.incl(adjacentNode)
    settledNodes.incl(currentNode)
  result = graph

proc main() =
  var nodeA = Node(name: "A")
  var nodeB = Node(name: "B")
  var nodeC = Node(name: "C")
  var nodeD = Node(name: "D")
  var nodeE = Node(name: "E")
  var nodeF = Node(name: "F")

  nodeA.addDestination(nodeB, 10)
  nodeA.addDestination(nodeC, 15)

  nodeB.addDestination(nodeD, 12)
  nodeB.addDestination(nodeF, 15)

  nodeC.addDestination(nodeE, 10)

  nodeD.addDestination(nodeE, 2)
  nodeD.addDestination(nodeF, 1)

  nodeF.addDestination(nodeE, 5)

  var graph = Graph()
  graph.addNode(nodeA)
  graph.addNode(nodeB)
  graph.addNode(nodeC)
  graph.addNode(nodeD)
  graph.addNode(nodeE)
  graph.addNode(nodeF)

  graph = calculateShortestPathFromSource(graph, nodeA)

  template toList(x: untyped): untyped = toDoublyLinkedList(x)

  var shortestPathForNodeB = toList [nodeA]
  var shortestPathForNodeC = toList [nodeA]
  var shortestPathForNodeD = toList [nodeA, nodeB]
  var shortestPathForNodeE = toList [nodeA, nodeB, nodeD]
  var shortestPathForNodeF = toList [nodeA, nodeB, nodeD]

  for node in graph.nodes:
    case node.name
    of "B":
      assert node.shortestPath == shortestPathForNodeB
    of "C":
      assert node.shortestPath == shortestPathForNodeC
    of "D":
      assert node.shortestPath == shortestPathForNodeD
    of "E":
      assert node.shortestPath == shortestPathForNodeE
    of "F":
      assert node.shortestPath == shortestPathForNodeF

main()
