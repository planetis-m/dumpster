
type
   Player = enum
      X = "X"
      O = "O"
      NA = " "

proc nextPlayer(p: Player): Player =
   assert p != NA
   Player(1 - int(p))

var p: Player
p = nextPlayer(p)

var f = proc (x: Player): Player = Player((int(p) + 1) mod 3)

for i in 1 .. 10:
   p = f(p)
   echo p
