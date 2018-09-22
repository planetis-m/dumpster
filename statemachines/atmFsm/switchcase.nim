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

proc main() =
   var eNextState = IdleState
   while true:
      # Read system Events
      let eNewEvent = readEvent()
      case eNextState
      of Idle:
         if CardInsert == eNewEvent:
            eNextState = insertCardHandler()
      of CardInserted:
         if PinEnter == eNewEvent:
            eNextState = enterPinHandler()
      of PinEntered:
         if OptionSelection == eNewEvent:
            eNextState = optionSelectionHandler()
      of OptionSelected:
         if AmountEnter == eNewEvent:
            eNextState = enterAmountHandler()
      of AmountEntered:
         if AmountDispatch == eNewEvent:
            eNextState = amountDispatchHandler()
      else:
         discard
