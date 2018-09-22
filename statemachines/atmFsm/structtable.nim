# https://aticleworld.com/state-machine-using-c/finite-state-in-c/
type
   # Different state of ATM machine
   SystemState = enum
      Idle,
      CardInserted,
      PinEntered,
      OptionSelected,
      AmountEntered,

   # Different type events
   SystemEvent = enum
      CardInsert,
      PinEnter,
      OptionSelection,
      AmountEnter,
      AmountDispatch

   # type alias of 2d array
   EventHandlerArray[N: static[int]] = array[N, StateRecord]

   # type alias of function pointer
   EventHandlerProc = proc (): SystemState {.nimcall.}

   # structure of state and event with event handler
   StateRecord = tuple
      eState: SystemState
      eEvent: SystemEvent
      pEventHandler: EventHandlerProc

# Prototype of eventhandlers
proc amountDispatchHandler(): SystemState =
   # function call to dispatch the amount and return the ideal state
   result = Idle

proc enterAmountHandler(): SystemState =
   # function call to Enter amount and return amount entered state
   result = AmountEntered

proc optionSelectionHandler(): SystemState =
   # function call to option select and return the option selected state
   result = OptionSelected

proc enterPinHandler(): SystemState =
   # function call to enter the pin and return pin entered state
   result = PinEntered

proc insertCardHandler(): SystemState =
   # function call to processing track data and return card inserted state
   result = CardInserted

# Initialize array of structure with states and event with proper handler
const asStateMachine: EventHandlerArray[5] = [
   (Idle, CardInsert, insertCardHandler),
   (CardInserted, PinEnter, enterPinHandler),
   (PinEntered, OptionSelection, optionSelectionHandler),
   (OptionSelected, AmountEnter, enterAmountHandler),
   (AmountEntered, AmountDispatch, amountDispatchHandler)
]

proc lookupTransition(eCurState: SystemState, eCurEvent: SystemEvent): EventHandlerProc =
   for i in 0 ..< asStateMachine.len:
      if asStateMachine[i].eState == eCurState and
            asStateMachine[i].eEvent == eCurEvent:
         return asStateMachine[i].pEventHandler
   result = nil

proc main() =
   var eNextState = Idle

   while true:
      # Read system Events
      let eNewEvent = readEvent()
      var pEventHandler: EventHandlerProc
      if eNextState <= SystemState.high and eNewEvent <= SystemEvent.high and
            (pEventHandler = lookupTransition(eNextState, eNewEvent); pEventHandler) != nil:
         # function call as per the state and event and return the next state of the finite state machine
         eNextState = pEventHandler()
      else:
         # Invalid
         discard
