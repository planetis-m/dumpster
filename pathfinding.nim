import sugar, deques, sets, algorithm
include tables

proc at[A, B](t: Table[A, B]; i: int): (A, B) =
  (t.data[i].key, t.data[i].val)

proc index[A, B](t: Table[A, B]; key: A): int =
  var hc: Hash
  result = rawGet(t, key, hc)

proc reversePath[N](parents: Table[N, int]; parent: int -> int;
    start: int): seq[N] =
  result = collect(newSeq):
    block:
      var
        i = start
        node: N
        value: int
      while i < parents.data.len:
        (node, value) = parents.at(i)
        i = parent(value)
        node
  result.reverse()

proc bfs*[N](start: N; successors: N -> seq[N]; success: N -> bool;
      checkFirst = true): seq[N] =
  if checkFirst and success(start):
    return @[start]
  var toSee = initDeque[int]()
  var parents = initTable[N, int]()
  parents[start] = high(int)
  toSee.addLast(parents.index(start))
  while toSee.len > 0:
    let i = toSee.popFirst()
    let (node, _) = parents.at(i)
    for successor in successors(node):
      if success(successor):
        result = reversePath(parents, p => p, i)
        result.add(successor)
        return
      if successor notin parents:
        parents[successor] = i
        toSee.addLast(parents.index(successor))
  # just return an empty path

when isMainModule:
  type
    Pos = tuple
      x, y: int

  func successors(self: Pos): seq[Pos] =
    let (x, y) = self
    @[(x+1, y+2), (x+1, y-2), (x-1, y+2), (x-1, y-2),
      (x+2, y+1), (x+2, y-1), (x-2, y+1), (x-2, y-1)]

  let goal = (1, 1) #(4, 6) ..and it explodes!
  let result = bfs((1, 1), successors, p => p == goal)
  assert result.len == 4
