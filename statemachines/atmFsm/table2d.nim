# https://aticleworld.com/state-machine-using-c/
import macros

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
   StateTransitionsArray = array[SystemState, array[SystemEvent, EventHandlerProc]]

   # type alias of function pointer
   EventHandlerProc = proc (): SystemState {.nimcall.}

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

macro sparse(table, data: untyped): untyped =
   template insert(table, state, event, handler): untyped =
      table[state][event] = handler
   result = newStmtlist()
   expectKind(data[0], nnkBracket)
   for n in data[0]:
      expectKind(n, nnkExprColonExpr)
      expectKind(n[1], nnkBracket)
      for m in n[1]:
         expectKind(m, nnkExprColonExpr)
         result.add getAst(insert(table, n[0], m[0], m[1]))

proc main() =
   var eNextState = Idle
   # Table to define valid states and event of finite state machine
   var stateTransitions: StateTransitionsArray

   sparse(stateTransitions): [
      Idle: [CardInsert: insertCardHandler],
      CardInserted: [PinEnter: enterPinHandler],
      PinEntered: [OptionSelection: optionSelectionHandler],
      OptionSelected: [AmountEnter: enterAmountHandler],
      AmountEntered: [AmountDispatch: amountDispatchHandler]
   ]

   while true:
      # Read system Events
      let eNewEvent = readEvent()
      # Check null pointer and array boundary
      if eNextState <= SystemState.high and eNewEvent <= SystemEvent.high and
            stateTransitions[eNextState][eNewEvent] != nil:
         #function call as per the state and event and return the next state of the finite state machine
         eNextState = stateTransitions[eNextState][eNewEvent]()
      else:
         # Invalid
         discard
