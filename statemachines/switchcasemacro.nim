# https://aticleworld.com/state-machine-using-c/
import macros

type
   # Different state of ATM machine
   SystemState = enum
      Idle,
      CardInserted,
      PinEntered,
      OptionSelected,
      AmountEntered

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

macro cases(state, event, table: untyped): untyped =
   result = newStmtlist()
   expectKind(table, nnkStmtList)
   # top level case state
   result.add nnkCaseStmt.newTree(state)
   for n in table:
      expectKind(n, nnkCall)
      expectLen(n, 2)
      expectKind(n[0], nnkPar)
      expectLen(n[0], 2)
      expectLen(n[1], 1)
      # of state kind with sublevel case event
      #if not eqIdent(result[0][^1][0], n[0][0]):
      result[0].add nnkOfBranch.newTree(n[0][0], nnkCaseStmt.newTree(event))
      # of event kind with next state assignment
      result[0][^1][1].add nnkOfBranch.newTree(n[0][1],
         nnkAsgn.newTree(state, nnkCall.newTree(n[1][0])))
      # sublevel else discard
      result[0][^1][1].add nnkElse.newTree(
         nnkDiscardStmt.newTree(newEmptyNode()))
   echo result.repr

# macro cases(state, event, table: untyped): untyped =
#    template declare(table, state, event, handler) =
#       var table: array[state, array[event, handler]]
#    template insert(table, state, event, handler) =
#       table[state][event] = handler
#    let declVar = getAst(declare(table, kind[0], kind[1], kind[2]))
#    let body = newStmtlist()
#    expectKind(data, nnkStmtList)
#    for n in data:
#       expectLen(n, 2)
#       expectKind(n[0], nnkPar)
#       expectLen(n[0], 2)
#       expectLen(n[1], 1)
#       body.add getAst(insert(table, n[0][0], n[0][1], n[1]))
#    result = nnkStmtListExpr.newTree(declVar, body, table)
#    echo result.repr

proc readEvent: SystemEvent = discard

proc main() =
   var eNextState = Idle

   while true:
      # Read system Events
      let eNewEvent = readEvent()

      cases(eNextState, eNewEvent):
         (Idle, CardInsert): insertCardHandler
         (CardInserted, PinEnter): enterPinHandler
         (PinEntered, OptionSelection): optionSelectionHandler
         (OptionSelected, AmountEnter): enterAmountHandler
         (AmountEntered, AmountDispatch): amountDispatchHandler
