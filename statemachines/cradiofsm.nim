# This code describes the state machine for a very basic car radio system.
# It is basically an infinite loop that reads incoming events.
# The state machine is only 2 states: radio mode, or CD mode.
# The event is either a mode change from radio to cd back and forth,
# or a go to next (next preset for radio or next track for CD).

type
   States = enum
      Radio, Cd

   Events = enum
      Mode, Next

proc main() =
   # Default state is radio  
   var state = Radio
   var stationNumber = 0
   var trackNumber = 0

   # Infinite loop
   while true:
      # Read the next incoming event. Usually this is a blocking function.
      let event = readEventFromMessageQueue()

      # Switch the state and the event to execute the right transition.
      case state
      of Radio:
         case event
         of Mode:
            # Change the state
            state = Cd
         of Next:
            # Increase the station number
            stationNumber.inc
      of Cd:
         case event
         of Mode:
            # Change the state
            state = Radio
         of Next
            # Go to the next track
            trackNumber.inc
