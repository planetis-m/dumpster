type
   MouseAction = object
      action: string

proc initMouseAction(action: string): MouseAction =
   result.action = action

proc `$`(m: MouseAction): string =
   m.action

proc `==`(a, b: MouseAction): bool =
   a.action == b.action

# Necessary when __cmp__ or __eq__ is defined
# in order to make this class usable as a
# dictionary key:

# proc hash(m: MouseAction): Hash =
#    hash(self.action)

# Static fields; an enumeration of instances:
const
   appears = initMouseAction("mouse appears")
   runsAway = initMouseAction("mouse runs away")
   enters = initMouseAction("mouse enters trap")
   escapes = initMouseAction("mouse escapes")
   trapped = initMouseAction("mouse trapped")
   removed = initMouseAction("mouse removed")

type
   State = ref object
      runImpl: proc () {.nimcall.}
      nextImpl: proc (input: MouseAction): State {.nimcall.}

# A State has an operation, and can be moved
# into the next State given an Input:

proc run(s: State) =
   assert s.runImpl != nil
   s.runImpl()

proc next(s: State; input: MouseAction): State =
   assert s.nextImpl != nil
   s.nextImpl(input)

type
   StateMachine = object
      currentState: State

# Takes a list of Inputs to move from State to
# State using a template method.

proc initStateMachine(initialState: State): StateMachine =
   result.currentState = initialState
   result.currentState.run()

# Template method:
proc runAll(s: var StateMachine, inputs: seq[MouseAction]) =
   for i in inputs:
      echo(i)
      s.currentState = s.currentState.next(i)
      s.currentState.run()

# State Machine pattern using 'if' statements
# to determine the next state.

proc runWaiting()
proc runLuring()
proc runTrapping()
proc runHolding()
proc nextWaiting(input: MouseAction): State
proc nextLuring(input: MouseAction): State
proc nextTrapping(input: MouseAction): State
proc nextHolding(input: MouseAction): State

proc newWaiting(): State =
   new(result)
   result.runImpl = runWaiting
   result.nextImpl = nextWaiting

proc newLuring(): State =
   new(result)
   result.runImpl = runLuring
   result.nextImpl = nextLuring

proc newTrapping(): State =
   new(result)
   result.runImpl = runTrapping
   result.nextImpl = nextTrapping

proc newHolding(): State =
   new(result)
   result.runImpl = runHolding
   result.nextImpl = nextHolding

# Static variable initialization:
let
   waiting = newWaiting()
   luring = newLuring()
   trapping = newTrapping()
   holding = newHolding()

proc runWaiting() =
   echo("Waiting: Broadcasting cheese smell")

proc nextWaiting(input: MouseAction): State =
   if input == appears:
      luring
   else:
      waiting

proc runLuring() =
   echo("Luring: Presenting Cheese, door open")

proc nextLuring(input: MouseAction): State =
   if input == runsAway:
      waiting
   elif input == enters:
      trapping
   else:
      luring

proc runTrapping() =
   echo("Trapping: Closing door")

proc nextTrapping(input: MouseAction): State =
   if input == escapes:
      waiting
   elif input == trapped:
      holding
   else:
      trapping

proc runHolding() =
   echo("Holding: Mouse caught")

proc nextHolding(input: MouseAction): State =
   if input == removed:
      waiting
   else:
      holding

# Driver program
import strutils

var moves: seq[MouseAction] = @[]

for action in readFile("MouseMoves.txt").splitLines:
   moves.add(initMouseAction(action))

var mouseTrap = initStateMachine(waiting)
mouseTrap.runAll(moves)
