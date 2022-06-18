import math, hashes
from tables import rightSize

# Placeholder constants
const
   free = -1
   dummy = -2
   none = -3
   pertubShift = 5
   growthFactor = 4

type
   KeyValuePair[A, B] = tuple[hcode: Hash, key: A, val: B]
   KeyValuePairSeq[A, B] = seq[KeyValuePair[A, B]]
   Dict*[A, B] = object ## Space efficient dictionary with fast iteration and cheap resizes.
      filled: int
      used: int
      indices: seq[int]
      data: KeyValuePairSeq[A, B]

iterator genProbes(hash: Hash; mask: int): int =
   # Same sequence of probes used in the current dictionary design
   var shiftedHash = abs(hash)
   var i = shiftedHash and mask
   yield i
   while true:
      i = int(5u * uint(i) + uint(shiftedHash) + 1u) and mask
      yield i
      shiftedHash = shiftedHash shr pertubShift

proc lookup[A, B](self: Dict[A, B]; key: A; hashvalue: Hash): (int, int) =
   # Same lookup logic as currently used in real dicts
   assert self.filled < len(self.indices)   # At least one open slot
   var freeslot = none
   for i in genProbes(hashvalue, len(self.indices)-1):
      var index = self.indices[i]
      if index == free:
         if freeslot == none: return (free, i)
         else: return (dummy, freeslot)
      elif index == dummy:
         if freeslot == none:
            freeslot = i
      elif self.data[index].hcode == hashvalue and
           self.data[index].key == key:
         return (index, i)

proc makeIndex(n: int): seq[int] =
   result = newSeq[int](n)
   for i in 0 ..< n:
      result[i] = free

proc resize[A, B](self: var Dict[A, B]; n: int) =
   ## Reindex the existing hash/key/value entries.
   ## Entries do not get moved, they only get new indices.
   ## No calls are made to hash() or __eq__().
   let n = nextPowerOfTwo(n)  # round-up to power-of-two
   setLen(self.indices, n)
   for i in 0 ..< self.indices.len:
      self.indices[i] = free
   for index in 0 ..< self.data.len:
      var res = 0
      for i in genProbes(self.data[index].hcode, n-1):
         res = i
         if self.indices[i] == free:
            break
      self.indices[res] = index
   self.filled = self.used

proc initDict*[A, B](initialSize = 64): Dict[A, B] =
   assert isPowerOfTwo(initialSize)
   result = Dict[A, B](indices: makeIndex(initialSize),
      data: newSeq[KeyValuePair[A, B]](initialSize))

proc clear*[A, B](self: var Dict[A, B]) =
   for i in 0 ..< self.indices.len:
      self.indices[i] = free
   for i in 0 ..< self.data.len:
      self.data[i].hcode = 0
      self.data[i].key = default(A)
      self.data[i].val = default(B)
   self.used = 0
   self.filled = 0 # used + dummies

proc `[]`*[A, B](self: Dict[A, B]; key: A): B =
   let hashvalue = hash(key)
   let (index, i) = self.lookup(key, hashvalue)
   if index < 0:
      raise newException(KeyError, "Key not found: " & $key)
   self.data[index].val

proc mustRehash(length, counter: int): bool {.inline.} =
   assert(length > counter)
   (length * 2 < counter * 3) or (length - counter < 4)

proc `[]=`*[A, B](self: var Dict[A, B]; key: A; value: B) =
   let hashvalue = hash(key)
   let (index, i) = self.lookup(key, hashvalue)
   if index < 0:
      self.indices[i] = self.used
      self.data.add (hashvalue, key, value)
      inc(self.used)
      if index == free:
         inc(self.filled)
         if mustRehash(len(self.indices), self.filled):
            self.resize(growthFactor * self.used)
   else:
      self.data[index].val = value

proc del*[A, B](self: var Dict[A, B]; key: A) =
   let hashvalue = hash(key)
   let (index, i) = self.lookup(key, hashvalue)
   if index < 0:
      raise newException(KeyError, "Key not found: " & $key)
   self.indices[i] = dummy
   dec(self.used)
   # If needed, swap with the lastmost entry to avoid leaving a "hole"
   if index != self.used:
      let lasthash = self.data[^1].hcode
      let lastkey = self.data[^1].key
      let (lastindex, j) = self.lookup(lastkey, lasthash)
      assert lastindex >= 0 and i != j
      self.indices[j] = index
   # Remove the entry at index by putting x[high(x)] into position
   self.data.del(index)

proc toDict*[A, B](pairs: openArray[(A, B)]): Dict[A, B] =
   result = initDict[A, B](rightSize(pairs.len))
   for key, value in items(pairs):
      result[key] = value

proc len*[A, B](self: Dict[A, B]): int =
   result = self.used

iterator keys*[A, B](self: Dict[A, B]): A =
   for (_, key, _) in self.data.items:
      yield key

iterator values*[A, B](self: Dict[A, B]): B =
   for (_, _, value) in self.data.items:
      yield value

iterator pairs*[A, B](self: Dict[A, B]): (A, B) =
   for (_, key, value) in self.data.items:
      yield (key, value)

proc contains*[A, B](self: Dict[A, B]; key: A): bool =
   let (index, i) = self.lookup(key, hash(key))
   index >= 0

proc get*[A, B](self: Dict[A, B]; key: A): B =
   let (index, i) = self.lookup(key, hash(key))
   if index >= 0:
      self.data[index].val

proc pop*[A, B](self: var Dict[A, B]): (A, B) =
   if len(self) == 0:
      raise newException(KeyError, "pop(): dictionary is empty")
   result.key = self.data[^1].key
   result.value = self.data[^1].val
   self.del(result.key)

proc `$`*[A, B](self: Dict[A, B]): string =
   if len(self) == 0:
      result = "{:}"
   else:
      result = "{"
      for key, value in self.pairs:
         if result.len > 1: result.add(", ")
         result.addQuoted(key)
         result.add(": ")
         result.addQuoted(value)
      result.add("}")

when isMainModule:
   import strutils

   proc showStructure[A, B](self: Dict[A, B]) =
      # Diagnostic method. Not part of the API.
      echo(repeat("=", 50))
      echo("Dictionary: ", self)
      echo("Indices: ", self.indices)
      for i in 0 ..< self.data.len:
         echo(i, " ", self.data[i])
      echo(repeat("-", 50))

   var d1 = {"timmy": "red", "barry": "green", "guido": "blue"}.toDict
   d1.showStructure()

   var d2 = {"martha": "cold", "helen": "heat", "stew": "cold",
             "heart": "heat", "kids": "breeze", "bloom": "bleach"}.toDict
   d2.showStructure()
