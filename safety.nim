import std/strutils

type
  Node {.acyclic.} = ref object
    name: string
    kids: seq[Node]
    parent {.cursor.}: Node

proc initParents(tree: Node) =
  for kid in items(tree.kids):
    kid.parent = tree
    initParents(kid)

func walk(tree: Node, action: proc (node: Node, depth: int), depth = 0) {.effectsOf: action.} =
  action(tree, depth)
  for kid in items(tree.kids):
    walk(kid, action, depth + 1)

proc print(tree: Node) =
  walk(tree, proc (node: Node, depth: int) =
    echo repeat(' ', 2 * depth), node.name
  )

proc calcTotalDepth(tree: Node): int =
  var total = 0
  walk(tree, proc (_: Node, depth: int) =
    total += depth
  )
  return total

proc process(intro: Node): Node =
  var tree = Node(name: "root", kids: @[
    intro,
    Node(name: "one", kids: @[
      Node(name: "two"),
      Node(name: "three"),
    ]),
    Node(name: "four"),
  ])
  initParents(tree)
  # Test pointer stability.
  var internalIntro = tree.kids[0]
  tree.kids.add(Node(name: "outro"))
  print(internalIntro)
  # Print and calculate.
  print(tree)
  var totalDepth = 0
  for i in 0 ..< 200_000:
    totalDepth += calcTotalDepth(tree)
  echo "Total depth: ", totalDepth
  return tree

proc main() =
  var intro = Node(name: "intro")
  var tree = process(intro)
  echo intro.parent.name

main()
