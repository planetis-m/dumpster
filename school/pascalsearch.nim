
proc find[T, S](a: T, item: S): int {.inline.} =
   ## Returns the first index of `item` in `a` or -1 if not found. This requires
   ## appropriate `items` and `==` operations to work.
   result = -1
   var found = false
   var i = 0
   while not found and i <= high(a):
      if a[i] == item:
         result = i
         found = true
      else:
         i.inc

proc tests() =
   var empty: array[0, int]
   assert find(empty, 2) == -1
   assert find([1], 2) == -1
   assert find([2], 2) == 0
   assert find([1, 3], 2) == -1
   assert find([1, 3, 2], 2) == 2

tests()
