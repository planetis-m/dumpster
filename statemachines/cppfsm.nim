type
   TurnStyleContext = ref object of RootObj

   TurnStyleState = ref object of RootObj

   UnlockedTurnStyleState = ref object of TurnStyleState

   LockedTurnStyleState = ref object of TurnStyleState

   TurnStyleContextFSM = ref object of TurnStyleContext
      unlocked: UnlockedTurnStyleState
      locked: LockedTurnStyleState
      itsState: TurnStyleState # private

proc lock(this: TurnStyleContext) =
   echo "the gate is locked"

proc unlock(this: TurnStyleContext) =
   echo "the gate is unlocked"

proc alarm(this: TurnStyleContext) =
   echo "Sound the alarms!"

proc thankYou(this: TurnStyleContext) =
   echo "thank you"

method coin(this: TurnStyleState; c: TurnStyleContextFSM) {.base.} = discard
method pass(this: TurnStyleState; c: TurnStyleContextFSM) {.base.} = discard

proc coin(this: TurnStyleContextFSM) = this.itsState.coin(this)
proc pass(this: TurnStyleContextFSM) = this.itsState.pass(this)

proc getState(this: TurnStyleContextFSM): TurnStyleState = this.itsState
proc setState(this: TurnStyleContextFSM; s: TurnStyleState) = this.itsState = s

method coin(this: LockedTurnStyleState; c: TurnStyleContextFSM) =
   c.setState(c.unlocked)
   c.unlock()

method pass(this: LockedTurnStyleState; c: TurnStyleContextFSM) =
   c.alarm()

method coin(this: UnlockedTurnStyleState; c: TurnStyleContextFSM) =
   c.thankYou()

method pass(this: UnlockedTurnStyleState; c: TurnStyleContextFSM) =
   c.setState(c.locked)
   c.lock()

proc main() =
   let fsm = TurnStyleContextFSM(locked: LockedTurnStyleState(),
                                 unlocked: UnlockedTurnStyleState()) 
   fsm.setState(fsm.locked) # initial state
   fsm.lock() # Make sure the gate is locked.

   for action in lines("vendingMachine.txt"):
      case action
      of "a coin has been dropped":
         fsm.coin()
      of "the user passes":
         fsm.pass()
      else:
         echo "invalid"

main()
