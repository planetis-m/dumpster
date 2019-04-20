# https://gamedevelopment.tutsplus.com/tutorials/finite-state-machines-theory-and-implementation--gamedev-11867
type
   Function = proc (ant: Ant)
   StackFsm = ref object
      stack: seq[Function]

   Ant = ref object
      position: Vector3D
      velocity: Vector3D
      brain: StackFsm

proc update(this: StackFsm; ant: Ant) =
   let currentStateFunction = getCurrentState()

   if currentStateFunction != nil:
      currentStateFunction(ant)

proc popState(this: StackFsm) =
   discard this.stack.pop()

proc pushState(this: StackFsm, state: Function) =
   if this.getCurrentState() != state:
      stack.push(state)

proc getCurrentState(this: StackFsm): Function =
   result = if this.stack.len > 0: this.stack[this.stack.len - 1] else: nil

proc findLeaf(this: Ant)
proc goHome(this: Ant)
proc runAway(this: Ant)

# Ant functions
proc newAnt(posX, posY: Number): Ant =
   result.position = newVector3D(posX, posY)
   result.velocity = newVector3D(-1, -1)
#    result.brain = newFSM()

   # Tell the brain to start looking for the leaf.
   result.brain.pushState(findLeaf)

# The "findLeaf" state.
# It makes the ant move towards the leaf.
proc findLeaf(this: Ant) =
   # Move the ant towards the leaf.
   this.velocity = newVector3D(Game.instance.leaf.x - this.position.x, Game.instance.leaf.y - this.position.y)

   if distance(Game.instance.leaf, this) <= 10:
      # The ant is extremelly close to the leaf, it's time
      # to go home.
      this.brain.popState() # removes "findLeaf" from the stack.
      this.brain.pushState(goHome) # push "goHome" state, making it the active state.

   if distance(Game.mouse, this) <= MouseThreatRadius:
      # Mouse cursor is threatening us. Let's run away!
      # The "runAway" state is pushed on top of "findLeaf", which means
      # the "findLeaf" state will be active again when "runAway" ends.
      this.brain.pushState(runAway)

# The "goHome" state.
# It makes the ant move towards its home.
proc goHome(this: Ant) =
   # Move the ant towards home
   this.velocity = newVector3D(Game.instance.home.x - this.position.x, Game.instance.home.y - this.position.y)

   if distance(Game.instance.home, this) <= 10:
      # The ant is home, let's find the leaf again.
      this.brain.popState() # removes "goHome" from the stack.
      this.brain.pushState(findLeaf) # push "findLeaf" state, making it the active state

   if distance(Game.mouse, this) <= MouseThreatRadius:
      # Mouse cursor is threatening us. Let's run away!
      # The "runAway" state is pushed on top of "goHome", which means
      # the "goHome" state will be active again when "runAway" ends.
      this.brain.pushState(runAway)

# The "runAway" state.
# It makes the ant run away from the mouse cursor.
proc runAway(this: Ant) =
   # Move the ant away from the mouse cursor
   this.velocity = newVector3D(this.position.x - Game.mouse.x, this.position.y - Game.mouse.y)

   # Is the mouse cursor still close?
   if distance(Game.mouse, this) > MouseThreatRadius:
      # No, the mouse cursor has gone away. Let's go back to the previously
      # active state.
      this.brain.popState()

proc update(this: Ant) =
   # Update the FSM controlling the "brain". It will invoke the currently
   # active state function: findLeaf(), goHome() or runAway().
   this.brain.update(this)

   # Apply the velocity vector to the position, making the ant move.
   this.moveBasedOnVelocity()
