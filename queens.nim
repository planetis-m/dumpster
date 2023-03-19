const
  N = 8

type
  TCol = array[1..N, bool]
  TMainDiag = array[2..N*2, bool]
  TSecDiag = array[1-N..N-1, bool]
  TQueensArr = array[1..N, int]

var
  Col: TCol # Array of columns
  MD: TMainDiag # Array of "main" diagonals
  SD: TSecDiag # Array of "secondary" diagonals
  Queen: TQueensArr # Array of queen coordinates
  Num = 1

proc print(QA: TQueensArr) =
  # The procedure displays on the chess field with placed queens on the screen
  echo "Number ", Num
  for i in 1..N:
    for j in 1..N:
      if Queen[i] == j:
        stdout.write('Q')
      else:
        stdout.write('.')
    stdout.write('\n')
  stdout.write('\n')
  inc Num

proc setQueen(i, j: int) =
  # The procedure places the queen at the coordinates [i,j] and sets false in the area of the queen's defeat
  Queen[i] = j
  Col[j] = false
  MD[i+j] = false
  SD[i-j] = false

proc removeQueen(i, j: int) =
  # The procedure removes the queen at the coordinates [i,j]
  Col[j] = true
  MD[i+j] = true
  SD[i-j] = true

proc tryQueen(i: int) =
  # Recursive procedure "tries" to put a queen in a free place from the area of ​​defeat
  for j in 1..N: # i - row, j - column
    if Col[j] and MD[i+j] and SD[i-j]:
      setQueen(i,j)
      if i < N:
        tryQueen(i+1) # If we haven't reached the 8th line, then we call recursively
      else:
        print(Queen)
      removeQueen(i,j)

proc cleanArrays =
  # Procedure resets arrays
  for i in 1..N:
    Col[i] = true
  for i in 2..N*2:
    MD[i] = true
  for i in 1-N..N-1:
    SD[i] = true

proc main =
  cleanArrays()
  tryQueen(1)

main()
