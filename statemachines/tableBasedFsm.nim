# Define the states and events. If your state machine program has multiple
# source files, you would probably want to put these definitions in an "include"
# file and #include it in each source file. This is because the action
# procedures need to update current_state, and so need access to the state
# definitions.

type
   States = enum
      State1, State2, State3, MaxStates

   Events = enum
      Event1, Event2, MaxEvents

var
   currentState: States
   newEvent: Events

# Provide the fuction prototypes for each action procedure. In a real
# program, you might have a separate source file for the action procedures of 
# each state.

proc actionS1E1()
proc actionS1E2()
proc actionS2E1()
proc actionS2E2()
proc actionS3E1()
proc actionS3E2()
proc getNewEvent(): Events

# Define the state/event lookup table. The state/event order must be the
# same as the enum definitions. Also, the arrays must be completely filled - 
# don't leave out any events/states. If a particular event should be ignored in 
# a particular state, just call a "do-nothing" function.

const stateTable = [
   [actionS1E1, actionS1E2], # procedures for state 1
   [actionS2E1, actionS2E2], # procedures for state 2
   [actionS3E1, actionS3E2]  # procedures for state 3
]

# This is the heart of the state machine - where you execute the proper 
# action procedure based on the new event you have to process and your current 
# state. It's important to make sure the new event and current state are 
# valid, because unlike "switch" statements, the lookup table method has no 
# "default" case to catch out-of-range values. With a lookup table, 
# out-of-range values cause the program to crash!

proc main() =
   let newEvent = getNewEvent() # get the next event to process
   if newEvent >= 0 and newEvent < MaxEvents and
         currentState >= 0 and currentState < maxStates:
      stateTable[currentState][newEvent]() # call the action procedure
   else:
      # invalid event/state - handle appropriately

# In an action procedure, you do whatever processing is required for the
# particular event in the particular state. Among other things, you might have
# to set a new state.

proc actionS1E1() =
   # do some processing here
   currentState = State2 # set new state, if necessary

proc actionS1E2() =  # other action procedures
proc actionS2E1() =
proc actionS2E2() =
proc actionS3E1() =
proc actionS3E2() =

# Return the next event to process - how this works depends on your application.

proc getNewEvent(): Events =
   result = Event1
