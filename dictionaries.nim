import hashes, strutils, sequtils

# ----
# Data
# ----

let keys = @["guido", "sarah", "barry", "rachel", "tim"]
let values1 = @["blue", "orange", "green", "yellow", "red"]
let values2 = @["austin", "dallas", "tuscon", "reno", "portland"]
let values3 = @["apple", "banana", "orange", "pear", "peach"]
let hashes1 = map(keys, hash)
let entries = zip(keys, values1)

var entries2 = newSeq[tuple[a: Hash, b, c: string]](5)
for i in 0 ..< 5:
   entries2[i] = (hashes1[i], keys[i], values1[i])

var comb_entries = newSeq[tuple[a: Hash, b, c, d, e: string]](5)
for i in 0 ..< 5:
   comb_entries[i] = (hashes1[i], keys[i], values1[i], values2[i], values3[i])

# -----
# Start
# -----

proc association_list() =
   echo(@[
      zip(keys, values1),
      zip(keys, values2),
      zip(keys, values3)
   ])

#association_list()

#[
@[@[(a: "guido", b: "blue"), (a: "sarah", b: "orange"), (a: "barry", b: "green"),
    (a: "rachel", b: "yellow"), (a: "tim", b: "red")],
  @[(a: "guido", b: "austin"), (a: "sarah", b: "dallas"), (a: "barry", b: "tuscon"),
    (a: "rachel", b: "reno"), (a: "tim", b: "portland")],
  @[(a: "guido", b: "apple"), (a: "sarah", b: "banana"), (a: "barry", b: "orange"),
    (a: "rachel", b: "pear"), (a: "tim", b: "peach")]]
]#

proc seperate_chaining(n: int) =
   var buckets = newSeqWith(n, newSeq[tuple[a, b: string]]())
   for pair in entries:
      let key = pair[0]
      let i = hash(key) mod n
      buckets[i].add(pair)
   echo buckets

#seperate_chaining(2)

#[
@[@[(a: "guido", b: "blue"), (a: "barry", b: "green"), (a: "rachel", b: "yellow")],
  @[(a: "sarah", b: "orange"), (a: "tim", b: "red")]]
]#

#seperate_chaining(4)

#[
@[@[(a: "guido", b: "blue"), (a: "barry", b: "green")],
  @[(a: "tim", b: "red")],
  @[(a: "rachel", b: "yellow")],
  @[(a: "sarah", b: "orange")]]
]#

#seperate_chaining(8)

#[
@[@[],
  @[(a: "tim", b: "red")],
  @[],
  @[(a: "sarah", b: "orange")],
  @[(a: "guido", b: "blue"), (a: "barry", b: "green")],
  @[],
  @[(a: "rachel", b: "yellow")],
  @[]]
]#

proc open_addressing_linear(n: int) =
   var table = newSeq[tuple[key, value: string]](n)
   for h, key, value in items(entries2):
      var i = h mod n
      while table[i] != (nil, nil):
         i = (i + 1) mod n
      table[i] = (key, value)
   echo table

#open_addressing_linear(8)

#[
@[(a: nil, b: nil),
  (a: "tim", b: "red"),
  (a: nil, b: nil),
  (a: "sarah", b: "orange"),
  (a: "guido", b: "blue"),
  (a: "barry", b: "green"),
  (a: "rachel", b: "yellow"),
  (a: nil, b: nil)]
]#

proc open_addressing_multihash(n: int) =
   var table = newSeq[tuple[h: Hash, key, value: string]](n)
   for h, key, value in items(entries2):
      var perturb = h
      var i = h mod n
      while table[i] != (0, nil, nil):
         echo key, " collided with ", table[i].key
         i = (5 * i + perturb + 1) mod n
         perturb = perturb shr 5
      table[i] = (h, key, value)
   echo table

#open_addressing_multihash(8)

#[
barry collided with guido
tim collided with barry

@[(h: 0, key: nil, value: nil),
  (h: 7471978867146070804, key: "barry", value: "green"),
  (h: 0, key: nil, value: nil),
  (h: 8744003157969062427, key: "sarah", value: "orange"),
  (h: 7651432089509955428, key: "guido", value: "blue"),
  (h: 0, key: nil, value: nil),
  (h: 36698807668559974, key: "rachel", value: "yellow"),
  (h: 37429862720514561, key: "tim", value: "red")]
]#

proc compact_and_ordered(n: int) =
   var table = newSeqWith(n, -1)
   for pos, entry in entries2:
      var perturb = entry[0]
      var i = perturb mod n
      while table[i] != -1:
         i = (5 * i + perturb + 1) mod n
         perturb = perturb shr 5
      table[i] = pos
   echo entries2
   echo table

compact_and_ordered(8)

#[
@[(a: 7651432089509955428, b: "guido", c: "blue", d: "austin", e: "apple"),
  (a: 8744003157969062427, b: "sarah", c: "orange", d: "dallas", e: "banana"),
  (a: 7471978867146070804, b: "barry", c: "green", d: "tuscon", e: "orange"),
  (a: 36698807668559974, b: "rachel", c: "yellow", d: "reno", e: "pear"),
  (a: 37429862720514561, b: "tim", c: "red", d: "portland", e: "peach")]

@[-1, 2, -1, 1, 0, -1, 3, 4]
]#

proc shared_and_compact(n: int) =
   var table = newSeqWith(n, -1)
   for pos, entry in entries2:
      var perturb = entry[0]
      var i = perturb mod n
      while table[i] != -1:
         i = (5 * i + perturb + 1) mod n
         perturb = perturb shr 5
      table[i] = pos
   echo comb_entries
   echo table

#shared_and_compact(16)

#[
@[(a: 7651432089509955428, b: "guido", c: "blue", d: "austin", e: "apple"),
  (a: 8744003157969062427, b: "sarah", c: "orange", d: "dallas", e: "banana"),
  (a: 7471978867146070804, b: "barry", c: "green", d: "tuscon", e: "orange"),
  (a: 36698807668559974, b: "rachel", c: "yellow", d: "reno", e: "pear"),
  (a: 37429862720514561, b: "tim", c: "red", d: "portland", e: "peach")]

@[-1, 4, -1, -1, 0, -1, 3, -1, -1, 2, -1, 1, -1, -1, -1, -1]
]#
