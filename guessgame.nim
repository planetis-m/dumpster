import random

type
   Game = object
      bounds: Slice[int]
      target: int
      over: bool
      guessed: seq[int]

proc initGame(number: int): Game =
   result.bounds = 0 .. 11
   result.target = number
   result.guessed = @[]

proc makeGuess(g: var Game; guess: int) =
   assert guess notin g.guessed
   if guess == g.target:
      g.over = true
   else:
      g.guessed.add(guess)

proc aiGuess(g: Game): int =
   result = rand(g.bounds)

proc main() =
   randomize()

   var game = initGame(3)
   while not game.over:
      let guess = aiGuess(game)
      echo guess
      game.makeGuess(guess)

main()
