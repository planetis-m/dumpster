# https://aticleworld.com/state-machine-using-c/finite-state-in-c/
import fusion/astdsl, macros

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

proc readEvent: SystemEvent = discard

macro cases(state, event, table: untyped): untyped =
  expectKind(table[0], nnkBracket)
  result = buildAst caseStmt(state):
    for n in table[0]:
      expectKind(n, nnkExprColonExpr)
      expectKind(n[1], nnkBracket)
      expectMinLen(n[1], 1)
      ofBranch(n[0]):
        ifStmt:
          for m in n[1]:
            expectKind(m, nnkExprColonExpr)
            elifBranch(call(ident"==", m[0], event)):
              asgn(state, call(m[1]))
          `else`:
            discardStmt(empty())

proc main() =
   var eNextState = Idle
   while true:
      # Read system Events
      let eNewEvent = readEvent()
      cases(eNextState, eNewEvent): [
        Idle: [CardInsert: insertCardHandler],
        CardInserted: [PinEnter: enterPinHandler],
        PinEntered: [OptionSelection: optionSelectionHandler],
        OptionSelected: [AmountEnter: enterAmountHandler],
        AmountEntered: [AmountDispatch: amountDispatchHandler]]

macro cases(state, event: untyped, table: varargs[untyped]): untyped =
  result = buildAst(stmtList):
    for n in table:
      expectKind(n, nnkOfBranch)
      expectKind(n[0], nnkPar)
      expectMinLen(n[1], 1)
      ifStmt:
        elifBranch(infix(ident"==", n[0][0], state)):
          ifStmt:
            elifBranch(infix(ident"==", n[0][1], event)):
              asgn(state, call(n[1]))

proc main() =
   var eNextState = Idle
   while true:
      # Read system Events
      let eNewEvent = readEvent()
      cases(eNextState, eNewEvent):
      of (Idle, CardInsert): insertCardHandler
      of (CardInserted, PinEnter): enterPinHandler
      of (PinEntered, OptionSelection): optionSelectionHandler
      of (OptionSelected, AmountEnter): enterAmountHandler
      of (AmountEntered, AmountDispatch): amountDispatchHandler
