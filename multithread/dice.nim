import std / [random, os, strformat], threadutils

const
  threadNum = 8

var
  p: array[threadNum, Thread[int]]
  diceValues: array[threadNum, int]
  status: array[threadNum, bool]

  barrierRolledDice: Barrier
  barrierCalculated: Barrier

proc roll(i: int) =
  while true:
    barrierEnter(barrierRolledDice)
    diceValues[i] = rand(1 .. 6)
    barrierLeave(barrierRolledDice)
    closeBarrier(barrierCalculated)

    sleep(1000)
    if status[i]:
      echo &"({i} rolled {diceValues[i]}) I won"
    else:
      echo &"({i} rolled {diceValues[i]}) I lost"
    openBarrier(barrierCalculated)

proc main =
  #randomize()

  for i in 0 ..< threadNum:
    createThread(p[i], roll, i)

  while true:
    closeBarrier(barrierRolledDice)
    # Calculate winner
    var maxRoll = diceValues[0]
    for i in 1 ..< threadNum:
      if diceValues[i] > maxRoll:
        maxRoll = diceValues[i]

    barrierEnter(barrierCalculated)
    for i in 0 ..< threadNum:
      status[i] = diceValues[i] == maxRoll
    barrierLeave(barrierCalculated)

    sleep(1000)
    echo("==== New round starting ====")
    openBarrier(barrierRolledDice)

  joinThreads(p)

main()
