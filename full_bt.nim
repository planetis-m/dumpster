# Checking if a binary tree is a full binary tree in Nim

type
  Node = ref object
    item: int
    leftChild: Node
    rightChild: Node

# Creating a node
proc newNode(item: int): Node =
  Node(item: item)

# Checking full binary tree
proc isFullTree(root: Node): bool =

  # Tree empty case
  if root == nil:
    return true

  # Checking whether child is present
  if root.leftChild == nil and root.rightChild == nil:
    return true

  if root.leftChild != nil and root.rightChild != nil:
    return isFullTree(root.leftChild) and isFullTree(root.rightChild)

  return false

proc main =
  var root = newNode(1)
  root.rightChild = newNode(3)
  root.leftChild = newNode(2)

  root.leftChild.leftChild = newNode(4)
  root.leftChild.rightChild = newNode(5)
  root.leftChild.rightChild.leftChild = newNode(6)
  root.leftChild.rightChild.rightChild = newNode(7)

  if isFullTree(root):
    echo "The tree is a full binary tree"
  else:
    echo "The tree is not a full binary tree"

main()
