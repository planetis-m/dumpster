import std/tables

type
   State = tuple[r: int, s: Transition]
   Transition = TableRef[(int, int), State]

# states
var
   p0c0: State
   p1c0: State
   p0c1: State
   p1c1: State

# transitions between states
p0c0 = (0, {(0, 0): p0c0, (1, 0): p1c0, (0, 1): p1c0, (1, 1): p0c1})
p1c0 = (1, {(0, 0): p0c0, (1, 0): p1c0, (0, 1): p1c0, (1, 1): p0c1})
p0c1 = (0, {(0, 0): p1c0, (1, 0): p0c1, (0, 1): p0c1, (1, 1): p1c1})
p1c1 = (1, {(0, 0): p1c0, (1, 0): p0c1, (0, 1): p0c1, (1, 1): p1c1})

proc add(x, y: string) =
   x = map(int, reversed(x))
   y = map(int, reversed(y))
   z = newSeq[int]()

   # simulate automaton
   value, transition = p0c0
   for r, s in zip_longest(x, y, fillvalue=0):
      value, transition = transition[r, s]
      z.append(value)

   # handle carry
   z.append(transition[0, 0][0])

   return ''.join(map(str, reversed(z)))
