import random, math

const
  objectCount = 1000000
  avoidCount = 20
  maxSpriteCount = 1100000

type
  SpriteData = object
    posX, posY: float
    scale: float
    colR, colG, colB: float
    sprite: float

# ----------------------------------------------------------------------------------
# components we use in our "game". these are all just simple structs with some data.
# ----------------------------------------------------------------------------------

type
  PositionComponent = object
    # 2D position: just x, y coordinates
    x, y: float

  SpriteComponent = object
    # Sprite: color, sprite index (in the sprite atlas), and scale for rendering it
    colorR, colorG, colorB: float
    spriteIndex: int
    scale: float

  WorldBoundsComponent = object
    # World bounds for our "game" logic: x,y minimum & maximum values
    xMin, xMax, yMin, yMax: float

  MotionComponent = object
    # ove around with constant velocity. When reached world bounds, reflect back from them.
    velx, vely: float

proc initialise(self: var MotionComponent, speedRange: Slice[float]) =
  # random angle
  let angle = rand(1.0) * Pi * 2
  # random movement speed between given min & max
  let speed = rand(speedRange)
  self.velx = cos(angle) * speed
  self.vely = sin(angle) * speed

# ------------------------------------------------------------------------
# super simple "game entities system", using struct-of-arrays data layout.
# we just have an array for each possible component, and a flags array bit
# bits indicating which components are "present".

# "ID" of a game object is just an index into the scene array.
type
  EntityId = int

  Tag = enum
    Position, Sprite, WorldBounds, Motion

  Entities = object
    # arrays of data; the sizes of all of them are the same. EntityID (just an index)
    # is used to access data for any "object/entity". The "object" itself is nothing
    # more than just an index into these arrays.

    # names of each object
    names: seq[string]
    # data for all components
    positions: seq[PositionComponent]
    sprites: seq[SpriteComponent]
    worldBounds: seq[WorldBoundsComponent]
    moves: seq[MotionComponent]
    # bit flags for every component, indicating whether this object "has it"
    flags: seq[set[Tag]]

proc initEntities(n: int): Entities =
  result = Entities(names: newSeqOfCap[string](n),
        positions: newSeqOfCap[PositionComponent](n),
        sprites: newSeqOfCap[SpriteComponent](n),
        worldBounds: newSeqOfCap[WorldBoundsComponent](n),
        moves: newSeqOfCap[MotionComponent](n),
        flags: newSeqOfCap[set[Tag]](n))

proc addEntity(self: var Entities, name: string): EntityId =
  result = self.names.len
  self.names.add(name)
  self.positions.add(PositionComponent())
  self.sprites.add(SpriteComponent())
  self.worldBounds.add(WorldBoundsComponent())
  self.moves.add(MotionComponent())
  self.flags.add({})

# The "scene"
var entities: Entities

# ------------------------------------------------------------------
# "systems" that we have; they operate on components of game objects

type
  MovementSystem = object
    boundsId: EntityId        # ID if object with world bounds
    objectList: seq[EntityId] # IDs of objects that should be moved

proc addObjectToSystem(self: var MovementSystem, id: EntityId) =
  self.objectList.add(id)

proc updateSystem(self: var MovementSystem, deltaTime: float) =
  template bounds: untyped = entities.worldBounds[self.boundsId]

  # go through all the objects
  for io in 0 ..< self.objectList.len:
    template pos: untyped = entities.positions[io]
    template move: untyped = entities.moves[io]

    # update position based on movement velocity & delta time
    pos.x += move.velx * deltaTime
    pos.y += move.vely * deltaTime

    # check against world bounds; put back onto bounds and mirror the velocity component to "bounce" back
    if pos.x < bounds.xMin:
      move.velx = -move.velx
      pos.x = bounds.xMin
    if pos.x > bounds.xMax:
      move.velx = -move.velx
      pos.x = bounds.xMax
    if pos.y < bounds.yMin:
      move.vely = -move.vely
      pos.y = bounds.yMin
    if pos.y > bounds.yMax:
      move.vely = -move.vely
      pos.y = bounds.yMax

var movementSystem: MovementSystem

# "Avoidance system" works out interactions between objects that "avoid" and "should be avoided".
# Objects that avoid:
# - when they get closer to things that should be avoided than the given distance, they bounce back,
# - also they take sprite color from the object they just bumped into

type
  AvoidanceSystem = object
    # things to be avoided: distances to them, and their IDs
    avoidDistanceList: seq[float]
    avoidList: seq[EntityId]
    # objects that avoid: their IDs
    objectList: seq[EntityId]

proc addAvoidThisObjectToSystem(self: var AvoidanceSystem, id: EntityId,
    distance: float) =
  self.avoidList.add(id)
  self.avoidDistanceList.add(distance * distance)

proc addObjectToSystem(self: var AvoidanceSystem, id: EntityId) =
  self.objectList.add(id)

proc distanceSq(a, b: PositionComponent): float =
  let dx = a.x - b.x
  let dy = a.y - b.y
  result = dx * dx + dy * dy

proc resolveCollision(id: EntityId, deltaTime: float) =
  template pos: untyped = entities.positions[id]
  template move: untyped = entities.moves[id]

  # flip velocity
  move.velx = -move.velx
  move.vely = -move.vely

  # move us out of collision, by moving just a tiny bit more than we'd normally move during a frame
  pos.x += move.velx * deltaTime * 1.1
  pos.y += move.vely * deltaTime * 1.1

proc updateSystem(self: var AvoidanceSystem, deltaTime: float) =
  # go through all the objects
  for io in 0 ..< self.objectList.len:
    let go = self.objectList[io]
    template myposition: untyped = entities.positions[go]

    # check each thing in avoid list
    for ia in 0 ..< self.avoidList.len:
      let avDistance = self.avoidDistanceList[ia]
      let avoid = self.avoidList[ia]
      template avoidposition: untyped = entities.positions[avoid]

      # is our position closer to "thing to avoid" position than the avoid distance?
      if distanceSq(myposition, avoidposition) < avDistance:
        resolveCollision(go, deltaTime)
        # also make our sprite take the color of the thing we just bumped into
        template avoidSprite: untyped = entities.sprites[avoid]
        template mySprite: untyped = entities.sprites[go]
        mySprite.colorR = avoidSprite.colorR
        mySprite.colorG = avoidSprite.colorG
        mySprite.colorB = avoidSprite.colorB

var avoidanceSystem: AvoidanceSystem

# --------
# the game
# --------

proc initialise() =
  entities = initEntities(1 + objectCount + avoidCount)

  # create "world bounds" object
  let go = entities.addEntity("bounds")
  template bounds: untyped = entities.worldBounds[go]
  bounds.xMin = -80.0
  bounds.xMax = 80.0
  bounds.yMin = -50.0
  bounds.yMax = 50.0
  entities.flags[go].incl WorldBounds
  movementSystem.boundsId = go

  # create regular objects that move
  for i in 0 ..< objectCount:
    let go = entities.addEntity("object")

    # position it within world bounds
    entities.positions[go].x = rand(bounds.xMin..bounds.xMax)
    entities.positions[go].y = rand(bounds.yMin..bounds.yMax)
    entities.flags[go].incl Position

    # setup a sprite for it (random sprite index from first 5), and initial white color
    entities.sprites[go].colorR = 1.0
    entities.sprites[go].colorG = 1.0
    entities.sprites[go].colorB = 1.0
    entities.sprites[go].spriteIndex = rand(0..4)
    entities.sprites[go].scale = 1.0
    entities.flags[go].incl Sprite

    # make it move
    entities.moves[go].initialise(0.5..0.7)
    entities.flags[go].incl Motion
    movementSystem.addObjectToSystem(go)

    # make it avoid the bubble things, by adding to the avoidance system
    avoidanceSystem.addObjectToSystem(go)

  # create objects that should be avoided
  for i in 0 ..< avoidCount:
    let go = entities.addEntity("toavoid")

    # position it in small area near center of world bounds
    entities.positions[go].x = rand(bounds.xMin..bounds.xMax) * 0.2
    entities.positions[go].y = rand(bounds.yMin..bounds.yMax) * 0.2
    entities.flags[go].incl Position

    # setup a sprite for it (6th one), and a random color
    entities.sprites[go].colorR = rand(0.5..1.0)
    entities.sprites[go].colorG = rand(0.5..1.0)
    entities.sprites[go].colorB = rand(0.5..1.0)
    entities.sprites[go].spriteIndex = 5
    entities.sprites[go].scale = 2.0
    entities.flags[go].incl Sprite

    # make it move, slowly
    entities.moves[go].initialise(0.1..0.2)
    entities.flags[go].incl Motion
    movementSystem.addObjectToSystem(go)

    # add to avoidance this as "Avoid This" object
    avoidanceSystem.addAvoidThisObjectToSystem(go, 1.3)

proc update(data: var seq[SpriteData], deltaTime: float): int =
  # returns amount of sprites
  # update object systems
  movementSystem.updateSystem(deltaTime)
  avoidanceSystem.updateSystem(deltaTime)

  # Using a smaller global scale "zooms out" the rendering, so to speak.
  const globalScale = 0.05
  # go through all objects
  for i in 0 ..< entities.flags.len:
    # For objects that have a Position & Sprite on them: write out
    # their data into destination buffer that will be rendered later on.
    if entities.flags[i] * {Sprite, Position} == {Sprite, Position}:
      template spr: untyped = data[result]
      template pos: untyped = entities.positions[i]
      spr.posX = pos.x * globalScale
      spr.posY = pos.y * globalScale
      template sprite: untyped = entities.sprites[i]
      spr.scale = sprite.scale * globalScale
      spr.colR = sprite.colorR
      spr.colG = sprite.colorG
      spr.colB = sprite.colorB
      spr.sprite = sprite.spriteIndex.float
      result.inc
