import math

type
  StackAllocationHeader = object
    size: int
    prev: pointer

proc alignup(x, align: uint): uint =
  result = (x + align - 1) and not(align - 1)

proc getPaddingWithHeader1(address: uint, alignment: int): int =
  let
    alignedAddress = alignup(address, alignment.uint)
    padding = alignedAddress - address
    neededSpace = sizeof(StackAllocationHeader).uint
  if padding < neededSpace:
    let numAlignments = (neededSpace - padding + alignment.uint - 1) div alignment.uint
    result = (numAlignments * alignment.uint + padding).int
  else:
    result = padding.int

proc getPaddingWithHeader2(address: uint, alignment: int): int =
  let
    neededSpace = sizeof(StackAllocationHeader).uint
    headerAddress = address + neededSpace
    alignedAddress = alignup(headerAddress, alignment.uint)
    padding = alignedAddress - address
  result = padding.int

proc compareResults(address: uint, alignment: int) =
  let
    result1 = getPaddingWithHeader1(address, alignment)
    result2 = getPaddingWithHeader2(address, alignment)
  echo "Address: ", address, ", Alignment: ", alignment
  echo "Result 1: ", result1
  echo "Result 2: ", result2
  echo "Same result: ", result1 == result2
  echo "---"
  if result1 != result2: quit("hee")

# Test cases
for i in 0..10000:
  compareResults(i.uint, 128)

# compareResults(100, 8)
# compareResults(105, 16)
# compareResults(200, 32)
# compareResults(1001, 64)
# compareResults(1023, 128)
