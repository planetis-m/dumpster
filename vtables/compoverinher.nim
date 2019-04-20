type
   IVisible = ref object of RootObj
      drawImpl: proc (this: IVisible) {.nimcall.}

   Invisible = ref object of IVisible

   Visible = ref object of IVisible

# ----------------------------------
# IVisible interface implementations
# ----------------------------------

proc draw(this: IVisible) =
   this.drawImpl(this)

proc drawInvisible(this: IVisible) =
   var this = Invisible(this)
   echo("I won't appear.")

proc newInvisible(): Invisible =
   new(result)
   result.drawImpl = drawInvisible

proc drawVisible(this: IVisible) =
   var this = Visible(this)
   echo("I'm showing myself.")

proc newVisible(): Visible =
   new(result)
   result.drawImpl = drawVisible

type
   ICollidable = ref object of RootObj
      collideImpl: proc (this: ICollidable) {.nimcall.}

   Solid = ref object of ICollidable

   NotSolid = ref object of ICollidable

# -------------------------------------
# ICollidable interface implementations
# -------------------------------------

proc collide(this: ICollidable) =
   this.collideImpl(this)

proc collideSolid(this: ICollidable) =
   var this = Solid(this)
   echo("Bang!")

proc newSolid(): Solid =
   new(result)
   result.collideImpl = collideSolid

proc collideNotSolid(this: ICollidable) =
   var this = NotSolid(this)
   echo("Splash!")

proc newNotSolid(): NotSolid =
   new(result)
   result.collideImpl = collideNotSolid

type
   IUpdatable = ref object of RootObj
      updateImpl: proc (this: IUpdatable) {.nimcall.}

   Movable = ref object of IUpdatable

   NotMovable = ref object of IUpdatable

# ------------------------------------
# IUpdatable interface implementations
# ------------------------------------

proc update(this: IUpdatable) =
   this.updateImpl(this)

proc updateMovable(this: IUpdatable) =
   var this = Movable(this)
   echo("Moving forward.")

proc newMovable(): Movable =
   new(result)
   result.updateImpl = updateMovable

proc updateNotMovable(this: IUpdatable) =
   var this = NotMovable(this)
   echo("I'm staying put.")

proc newNotMovable(): NotMovable =
   new(result)
   result.updateImpl = updateNotMovable

type
   GameObject = ref object of RootObj
      v: IVisible
      u: IUpdatable
      c: ICollidable

# ------------------------------
# GameObject type implementation
# ------------------------------

template newGameObject(visible: IVisible, updatable: IUpdatable, collidable: ICollidable) =
   new(result)
   result.v = visible
   result.u = updatable
   result.c = collidable

proc update(this: GameObject) =
   this.u.update()

proc draw(this: GameObject) =
   this.v.draw()

proc collide(this: GameObject) =
   this.c.collide()

type
   Player = ref object of GameObject

   Cloud = ref object of GameObject

   Building = ref object of GameObject

   Trap = ref object of GameObject

proc newPlayer(): Player =
   newGameObject(newVisible(), newMovable(), newSolid())

proc newCloud(): Cloud =
   newGameObject(newVisible(), newMovable(), newNotSolid())

proc newBuilding(): Building =
   newGameObject(newVisible(), newNotMovable(), newSolid())

proc newTrap(): Trap =
   newGameObject(newInvisible(), newNotMovable(), newSolid())

proc main() =
   let player = newPlayer()
   player.update()
   player.collide()
   player.draw()

main()
