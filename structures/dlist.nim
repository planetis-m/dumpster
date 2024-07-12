type
  DoublyLinkedNode = ref object
    next: DoublyLinkedNode
    prev: DoublyLinkedNode
    value: int

  DoublyLinkedList = object
    ## A doubly linked list.
    head: DoublyLinkedNode
    tail: DoublyLinkedNode

proc newDoublyLinkedNode(value: int): DoublyLinkedNode =
  result = DoublyLinkedNode(value: value)

proc insertBefore(L: var DoublyLinkedList, x, y: DoublyLinkedNode) =
  ## Insert node x before node y in the list L.
  ## If y is not in the list, x is not inserted.
  if y == nil:
    return

  if y == L.head:
    # Inserting at the beginning of the list
    x.next = L.head
    L.head.prev = x
    L.head = x
  else:
    # Inserting in the middle or end of the list
    x.prev = y.prev
    x.next = y
    if y.prev != nil:
      y.prev.next = x
    y.prev = x

  # Update tail if necessary
  if y == L.tail:
    L.tail = x.next

# Helper function to print the list (for testing purposes)
proc printList(L: DoublyLinkedList) =
  var current = L.head
  while current != nil:
    stdout.write($current.value & " <-> ")
    current = current.next
  echo "nil"

# Test the implementation
var L = DoublyLinkedList()
var node1 = newDoublyLinkedNode(1)
var node2 = newDoublyLinkedNode(2)
var node3 = newDoublyLinkedNode(3)

L.head = node1
L.tail = node3
node1.next = node2
node2.prev = node1
node2.next = node3
node3.prev = node2

echo "Original list:"
printList(L)

var newNode = newDoublyLinkedNode(4)
insertBefore(L, newNode, node2)

echo "After inserting 4 before 2:"
printList(L)

newNode = newDoublyLinkedNode(5)
insertBefore(L, newNode, L.head)

echo "After inserting 5 at the beginning:"
printList(L)

newNode = newDoublyLinkedNode(6)
insertBefore(L, newNode, L.tail)

echo "After inserting 6 before the tail:"
printList(L)
