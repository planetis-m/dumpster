import std / [sets, sugar, strformat]
import vertexedge

type
  EdgeList[T] = object
    vertex: Vertex[T]
    edges: seq[Edge[T]]

proc addEdge[T](self: var EdgeList[T]; edge: Edge[T]) =
  self.edges.add(edge)

type
  AdjacencyListGraph[T] = object
    adjacencyList: seq[EdgeList[T]]

proc vertices[T](self: AdjacencyListGraph[T]): seq[Vertex[T]] =
  result = collect(newSeq):
    for edgeList in self.adjacencyList.items:
      edgeList.vertex

proc edges[T](self: AdjacencyListGraph[T]): seq[Edge[T]] =
  let allEdges = collect(initHashSet):
    for edgeList in self.adjacencyList.items:
      for edge in edgeList.edges.items:
        edge
  result = collect(newSeq):
    for edge in allEdges.items:
      edge

proc createVertex[T](self: var AdjacencyListGraph[T]; data: T): Vertex[T] =
  # check if the vertex already exists
  let matchingVertices = collect(newSeq):
    for vertex in self.vertices.items:
      if vertex.data == data:
        vertex

  if matchingVertices.len > 0:
    result = matchingVertices[^1]
  else:
    # if the vertex doesn't exist, create a new one
    result = Vertex[T](data: data, index: self.adjacencyList.len)
    self.adjacencyList.add(EdgeList[T](vertex: result))

proc addDirectedEdge[T](self: var AdjacencyListGraph[T]; fr, to: Vertex[T];
    weight: float) =
  let edge = Edge[T](fr: fr, to: to, weight: weight)
  self.adjacencyList[fr.index].addEdge(edge)

proc addUndirectedEdge[T](self: var AdjacencyListGraph[T]; vertices: (Vertex[T],
    Vertex[T]); weight: float) =
  self.addDirectedEdge(vertices[0], to = vertices[1], withWeight = weight)
  self.addDirectedEdge(vertices[1], to = vertices[0], withWeight = weight)

proc weightFrom[T](self: AdjacencyListGraph[T]; sourceVertex,
    destinationVertex: Vertex[T]): float =
  for edge in self.adjacencyList[sourceVertex.index].edges:
    if edge.to == destinationVertex:
      return edge.weight
  result = 0.0

proc edgesFrom[T](self: AdjacencyListGraph[T]; sourceVertex: Vertex[T]): seq[Edge[T]] =
  self.adjacencyList[sourceVertex.index].edges

proc `$`[T](self: AdjacencyListGraph[T]): string =
  for edgeList in self.adjacencyList.items:
    if result.len > 0: result.add("\n")
    var row = ""
    for edge in edgeList.edges.items:
      if row.len > 0: row.add(", ")
      row.add($edge.to.data)
      if edge.weight > 0.0:
        row.add(&": {$edge.weight}")
    result.add(&"{$edgeList.vertex.data} -> [{row}]")

when isMainModule:
  var graph: AdjacencyListGraph[string]
  var nodeA = graph.createVertex("a")
  var nodeB = graph.createVertex("b")
  var nodeC = graph.createVertex("c")
  var nodeD = graph.createVertex("d")
  var nodeE = graph.createVertex("e")
  var nodeF = graph.createVertex("f")
  var nodeG = graph.createVertex("g")
  var nodeH = graph.createVertex("h")

  graph.addDirectedEdge(nodeA, to = nodeB, 1.0)
  graph.addDirectedEdge(nodeA, to = nodeC, 1.0)
  graph.addDirectedEdge(nodeB, to = nodeD, 1.0)
  graph.addDirectedEdge(nodeB, to = nodeE, 1.0)
  graph.addDirectedEdge(nodeC, to = nodeF, 1.0)
  graph.addDirectedEdge(nodeC, to = nodeG, 1.0)
  graph.addDirectedEdge(nodeE, to = nodeH, 1.0)
  graph.addDirectedEdge(nodeE, to = nodeF, 1.0)
  graph.addDirectedEdge(nodeF, to = nodeG, 1.0)
  echo graph
  echo graph.edgesFrom(nodeE)
