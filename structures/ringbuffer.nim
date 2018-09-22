
type
   Buffer*[T] = object
      head, tail: int
      cap, len: int
      data: ref UncheckedArray[T]

proc initBuffer*[T](cap: Natural): Buffer[T] =
   new(result.data)
   result.tail = -1
   result.cap = cap

template adjustHead(b) =
   b.head = (b.cap + b.tail - b.len + 1) mod b.cap

template adjustTail(b, i) =
   b.tail = (b.tail + i) mod b.cap

template emptyCheck(b) =
   # Bounds check for the regular buffer access.
   when compileOption("boundChecks"):
      if unlikely(b.len < 1):
         raise newException(IndexError, "Empty buffer.")

template xBoundsCheck(b, i) =
   # Bounds check for the array like accesses.
   when compileOption("boundChecks"):  # d:release should disable this.
      if unlikely(i >= b.len):  # x < b.low is taken care by the Natural parameter
         raise newException(IndexError,
            "Out of bounds: " & $i & " > " & $(b.len - 1))

proc add*[T](b: var Buffer[T], item: T) =
   ## Add an element to the buffer
   adjustTail(b, 1)
   b.data[b.tail] = item
   b.len = min(b.len + 1, b.cap)
   adjustHead(b)

proc add*[T](b: var Buffer[T], c: openArray[T]) =
   for item in c:
      adjustTail(b, 1)
      b.data[b.tail] = item
   b.len = min(b.len + len(c), b.cap)
   adjustHead(b)

proc `[]`*[T](b: Buffer[T], i: Natural): T {.inline.} =
   xBoundsCheck(b, i)
   b.data[(i + b.head) mod b.cap]

proc `[]`*[T](b: Buffer[T], x: Slice[int]): seq[T] =
   ## slice operation for bufers. returns the inclusive range [a[x.a], a[x.b]]:
   var L = x.b - x.a + 1
   newSeq(result, L)
   for i in 0 ..< L:
      result[i] = b[x.a + i]

proc `[]=`*[T](b: var Buffer[T], i: Natural, x: T) {.raises: [IndexError].} =
   ## Set an item at index (adjusted)
   xBoundsCheck(b, i)
   b.data[(i + b.head) mod b.cap] = x

proc `[]=`*[T](b: Buffer[T], x: Slice[int], c: openArray[T]): seq[T] =
   ## Create a subsequence of the buffer from elements s to e
   ## Creates a sequence of the entire collection by default.
   var L = x.b - x.a + 1
   if L == b.len:
      for i in 0 ..< L:
         b[x.a + i] = c[i]
   else:
      raise newException(RangeError, "different lengths for slice assignment")

iterator items*[T](b: Buffer[T]): T =
   var i = 0
   while i < len(b):
      yield b[i]
      inc(i)

iterator pairs*[T](b: Buffer[T]): tuple[a: int, b: T] =
   var i = 0
   while i < len(b):
      yield (i, b[i])
      inc(i)

proc pop*[T](b: var Buffer[T]): T =
   ## Remove an element from the buffer and return it
   emptyCheck(b)
   result = b.data[b.tail]
   adjustTail(b, -1)
   dec(b.len)
   adjustHead(b)

proc peekFirst*[T](b: Buffer[T]): T {.inline.} =
   ## Returns the first element of `b`, but does not remove it from the buffer.
   emptyCheck(b)
   result = b.data[b.head]

proc peekLast*[T](b: Buffer[T]): T {.inline.} =
   ## Returns the last element of `b`, but does not remove it from the buffer.
   emptyCheck(b)
   result = b.data[(b.tail - 1) and b.mask]

proc isFull*(b: Buffer): bool =
   ## Is the buffer at capacity (add will overwrite another element)
   b.len == b.cap

proc `@`*[T](b: Buffer[T]): seq[T] {.inline.} =
   ## Convert the buffer to a sequence
   b[0 .. b.len-1]

proc len*(b: Buffer): int = b.len

when isMainModule:
   var b = initBuffer[int](5)

   b.add([1, 2, 3, 4, 5])
   b.add(6)
   b.add([7, 8])

   assert @b == @[4, 5, 6, 7, 8]
