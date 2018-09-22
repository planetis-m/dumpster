proc moveDisk(fr, to: string) {.inline.} =
   echo "move from ", fr, " to ", to

proc hanoi(n: int, fr, to, spare: string) =
   if n == 1:
      moveDisk(fr, to)
   else:
      hanoi(n - 1, fr, spare, to) #1 [recursively] move N-1 disks from left to middle
      hanoi(1, fr, to, spare)     #2 move largest disk from left to right
      hanoi(n - 1, spare, to, fr) #3 [recursively] move N-1 disks from middle to right

hanoi(4, "left", "right", "middle")
