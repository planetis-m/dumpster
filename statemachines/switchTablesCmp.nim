# https://yakking.branchable.com/posts/state-machines-in-c/
type
   States = enum
      Start, Loop, End

   Events = enum
      StartLooping, PrintHello, StopLooping

var state = Start

proc main() =
   stepState(StartLooping)
   stepState(PrintHello)
   stepState(PrintHello)
   stepState(StopLooping)

# Handling states with case statements
proc stepState(event: Events) =
   case state
   of Start:
      case event
      of StartLooping:
         state = Loop
      else:
         echo(event, " is invalid for Start state")
   of Loop:
      case event
      of PrintHello:
         echo("Hello World!")
      of StopLooping:
         state = End
      else:
         echo(event, " is invalid for Loop state")
   of End:
      echo(event, " is invalid for End state")

# Handling states with a table
type
   EventHandler = proc (state: States, event: Events): States

proc doNothing(state: States, event: Events): States =
   result = state

proc startLooping(state: States, event: Events): States =
   result = Loop

proc printHello(state: States, event: Events): States =
   echo("Hello World!")
   result = Loop

proc stopLooping(state: States, event: Events): States =
   result = End

let transitions = [
   Start: [
      StartLooping: startLooping,
      PrintHello: doNothing,
      StopLooping: doNothing
   ], Loop: [
      StartLooping: doNothing,
      PrintHello: printHello,
      StopLooping: stopLooping
   ], End: [
      StartLooping: doNothing,
      PrintHello: doNothing,
      StopLooping: doNothing
   ]
]

proc stepState(event: Events) =
   var handler: EventHandler
   if event >= 0 and event < Events.len and
         state >= 0 and state < States.len:
      handler = transitions[state][event]
   state = handler(state, event)
