import strutils

proc hanoi(numOfDisks: Natural) =
   proc topDiskSize(rod: seq[int]): int {.inline.} =
      if rod.len == 0:
         return high(int)
      rod[rod.len - 1]

   template moveDisk(fr, to) =
      let disk = rods[fr].pop
      echo "move disk ", disk, " from ", moveStr[fr], " to ", moveStr[to]
      rods[to].add(disk)

   const moveStr = ["left", "middle", "right"]
   # To avoid using mod operator we precompute next and prev tables
   const flow = [(1, 2), (0, 2), (0, 1)]
   # Create three stacks of size 'numOfDisks' to hold the disks
   var rods = [newSeq[int](), newSeq[int](), newSeq[int]()]
   # push disks on our virtual rod's
   for i in countdown(numOfDisks, 1):
      rods[0].add(i)
   # we need (2 ^ numOfDisks - 1) moves
   let totalNumOfMoves = (1 shl numOfDisks) - 1
   # direction of rotation of the smallest disk
   let dir = if numOfDisks mod 2 == 0: -1 else: 1
   # move will point to the next legal movement
   var move = 0
   for i in 1 .. totalNumOfMoves:
      # in Nim -1 mod 3 == -1, we add 3 to get positive result
      move = (move + dir + 3) mod 3
      var (fr, to) = flow[move]
      # from is the rod with the smallest disk
      if rods[fr].topDiskSize > rods[to].topDiskSize:
         swap(fr, to)
      # move last disk from source to destination
      moveDisk(fr, to)

hanoi(4)
