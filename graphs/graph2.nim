import std / [sugar, strformat], vertexedge

type
  AdjacencyMatrixGraph[T] = object
    # If adjacencyMatrix[i][j] is not nil, then there is an edge from
    # vertex i to vertex j.
    adjacencyMatrix: seq[seq[float]]
    vertices: seq[Vertex[T]]

proc edges[T](self: AdjacencyMatrixGraph[T]): seq[Edge[T]] =
  result = collect(newSeq):
    for row in 0 ..< self.adjacencyMatrix.len:
      for column in 0 ..< self.adjacencyMatrix.len:
        let weight = self.adjacencyMatrix[row][column]
        if weight > 0:
          Edge(fr: vertices[row], to: vertices[column], weight: weight)

proc createVertex[T](self: var AdjacencyMatrixGraph[T]; data: T): Vertex[T] =
  ## Adds a new vertex to the matrix.
  ## Performance: possibly O(n^2) because of the resizing of the matrix.
  # check if the vertex already exists
  let matchingVertices = collect(newSeq):
    for vertex in self.vertices.items:
      if vertex.data == data:
        vertex

  if matchingVertices.len > 0:
    result = matchingVertices[^1]
  else:
    # if the vertex doesn't exist, create a new one
    result = Vertex[T](data: data, index: self.adjacencyMatrix.len)

    # Expand each existing row to the right one column.
    for i in 0 ..< self.adjacencyMatrix.len:
      self.adjacencyMatrix[i].add(0.0)

    # Add one new row at the bottom.
    let newRow = newSeq[float](self.adjacencyMatrix.len + 1)
    self.adjacencyMatrix.add(newRow)

    self.vertices.add(result)

proc addDirectedEdge[T](self: var AdjacencyMatrixGraph[T]; fr, to: Vertex[T];
    weight: float) =
  self.adjacencyMatrix[fr.index][to.index] = weight

proc addUndirectedEdge[T](self: var AdjacencyMatrixGraph[T]; vertices: (Vertex[
    T], Vertex[T]); weight: float) =
  self.addDirectedEdge(vertices[0], to = vertices[1], withWeight = weight)
  self.addDirectedEdge(vertices[1], to = vertices[0], withWeight = weight)

proc weightFrom[T](self: AdjacencyMatrixGraph[T]; sourceVertex,
    destinationVertex: Vertex[T]): float =
  self.adjacencyMatrix[sourceVertex.index][destinationVertex.index]

proc edgesFrom[T](self: AdjacencyMatrixGraph[T]; sourceVertex: Vertex[T]): seq[Edge[T]] =
  let fromIndex = sourceVertex.index
  result = collect(newSeq):
    for column in 0 ..< self.adjacencyMatrix.len:
      let weight = self.adjacencyMatrix[fromIndex][column]
      if weight > 0:
        Edge(fr: sourceVertex, to: vertices[column], weight: weight)

proc `$`[T](self: AdjacencyMatrixGraph[T]): string =
  let n = self.adjacencyMatrix.len
  for i in 0 ..< n:
    if result.len > 0: result.add("\n")
    var row = ""
    for j in 0 ..< n:
      let value = self.adjacencyMatrix[i][j]
      if value > 0:
        row.add(&"{value:<5.1f}")
      else:
        row.add("  ø  ")
    result.add(row)

when isMainModule:
  var graph: AdjacencyMatrixGraph[string]
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
