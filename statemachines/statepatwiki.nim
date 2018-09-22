import strutils

type
   Statelike = ref object of RootObj

   StateLowerCase = ref object of Statelike

   StateMultipleUpperCase = ref object of Statelike
      count: int # Counter local to this state

   StateContext = ref object
      myState: Statelike

using
   context: StateContext
   name: string

proc setState(this: StateContext, newState: Statelike) =
   # Normally only called by classes implementing the State interface.
   this.myState = newState

method writeName(this: Statelike, context, name) {.base.} = discard

method writeName(this: StateLowerCase, context, name) =
   echo(name.toLowerAscii)
   context.setState(StateMultipleUpperCase())

method writeName(this: StateMultipleUpperCase, context, name) =
   echo(name.toUpperAscii)
   # Change state after StateMultipleUpperCase's writeName() gets invoked twice
   this.count.inc
   if this.count > 1:
      context.setState(StateLowerCase())

proc writeName(this: StateContext, name) =
   this.myState.writeName(this, name)

proc main() =
   let sc = StateContext()
   sc.setState(StateLowerCase())

   sc.writeName("Monday")
   sc.writeName("Tuesday")
   sc.writeName("Wednesday")
   sc.writeName("Thursday")
   sc.writeName("Friday")
   sc.writeName("Saturday")
   sc.writeName("Sunday")

main()
