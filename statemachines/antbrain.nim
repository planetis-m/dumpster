# https://gamedevelopment.tutsplus.com/tutorials/finite-state-machines-theory-and-implementation--gamedev-11867
type
   StateMachine = ref object
      currentState: proc (ant: Ant) # points to the currently current state function

   Ant = ref object
      position: Vector3D
      velocity: Vector3D
      brain: StateMachine

proc changeState(state: proc(ant: Ant)) =
   this.currentState = state

proc update(this: StateMachine; ant: Ant) =
   if this.currentState != nil:
      this.currentState(ant)

proc findLeaf(this: Ant)
proc goHome(this: Ant)
proc runAway(this: Ant)

# Ant functions
proc newAnt(posX, posY: Number): Ant =
   result.position = newVector3D(posX, posY)
   result.velocity = newVector3D(-1, -1)
   result.brain = newFSM()

   # Tell the brain to start looking for the leaf.
   brain.changeState(findLeaf)

# The "findLeaf" state.
# It makes the ant move towards the leaf.
proc findLeaf(this: Ant) =
   # Move the ant towards the leaf.
   this.velocity = newVector3D(Game.instance.leaf.x - this.position.x, Game.instance.leaf.y - this.position.y)

   if distance(Game.instance.leaf, this) <= 10:
      # The ant is extremelly close to the leaf, it's time
      # to go home.
      this.brain.changeState(goHome)

   if distance(Game.mouse, this) <= MouseThreatRadius:
      # Mouse cursor is threatening us. Let's run away!
      # It will make the brain start calling runAway() from
      # now on.
      this.brain.changeState(runAway)

# The "goHome" state.
# It makes the ant move towards its home.
proc goHome(this: Ant) =
   # Move the ant towards home
   this.velocity = newVector3D(Game.instance.home.x - this.position.x, Game.instance.home.y - this.position.y)

   if distance(Game.instance.home, this) <= 10:
      # The ant is home, let's find the leaf again.
      this.brain.changeState(findLeaf)

# The "runAway" state.
# It makes the ant run away from the mouse cursor.
proc runAway(this: Ant) =
   # Move the ant away from the mouse cursor
   this.velocity = newVector3D(this.position.x - Game.mouse.x, this.position.y - Game.mouse.y)

   # Is the mouse cursor still close?
   if distance(Game.mouse, this) > MouseThreatRadius:
      # No, the mouse cursor has gone away. Let's go back looking for the leaf.
      this.brain.changeState(findLeaf)

proc update(this: Ant) =
   # Update the FSM controlling the "brain". It will invoke the currently
   # active state function: findLeaf(), goHome() or runAway().
   this.brain.update(this)

   # Apply the velocity vector to the position, making the ant move.
   this.moveBasedOnVelocity()
