# https://pixelmatic.github.io/articles/2020/05/13/ecs-and-ai.html
import math, fpenv

proc exp(x: float32, power = 2'f32): float32 =
  result = clamp(pow(x, power), 0'f32, 1'f32)

proc linear(x: float32, slope = 1'f32): float32 =
  result = clamp(x * slope, 0'f32, 1'f32)

proc decay(t, mag: float32): float32 =
  result = clamp(pow(mag, t), 0'f32, 1'f32)

proc sigmoid(t, k: float32): float32 =
  result = clamp(k * t / (k - t + 1'f32), 0'f32, 1'f32)

proc raiseFastToSlow(t: float32, k = 4'f32): float32 =
  result = clamp(-pow(t - 1'f32, k) + 1'f32, 0'f32, 1'f32)

proc createCat(world: var World, parent: Entity): Entity =
  result = world.build(blueprint):
    with:
      Cat()
      Hungry(value: 0)
      Tired(value: 0)
      Decision(action: None)
      EatScore()
      SleepScore()
      PlayScore()

type
  HasComponent = enum
    HasCat, HasDecision, HasEatAction, HasPlayScore, HasSleepScore,
    HasEatAction, HasPlayAction, HasSleepAction, HasHunger, HasTired

  Action = enum
    None, Eat, Sleep, Play

  EatAction = object
    hungerRecoverPerTick: float32
    tirednessCostPerTick: float32 # eat still get tired, but should slower than play

  SleepAction = object
    tirednessRecoverPerTick: float32
    hungerCostPerTick: float32 # sleep still get hungry slowly

  PlayAction = object
    # the hunger cost and tiredness cost of play are faster than eat and sleep
    hungerCostPerTick: float32
    tirednessCostPerTick: float32

  EatScore = object
    score: float32

  SleepScore = object
    score: float32

  PlayScore = object
    score: float32

  Hungry = object
    value: float32 # 0: not hungry, 100: hungry to death

  Tired = object
    value: float32 # 0: not tired, 100: tired to death

  Decision = object
    action: Action # current action to perform

const Query = {HasDecision, HasEatScore, HasSleepScore, HasPlayScore}

proc sysActionSelection =
  for entity, signature in game.world.signature.pairs:
    if signature * Query == Query:
      update(game, entity)

proc update(game: var Game, entity: Entity) =
  # Choose action base on the highest score
  template decision: untyped = game.world.decision[entity.idx]
  template eatScore: untyped = game.world.eatScore[entity.idx]
  template sleepScore: untyped = game.world.sleepScore[entity.idx]
  template playScore: untyped = game.world.playScore[entity.idx]

  var highestScore = 0'f32
  var actionToDo = Play
  if eatScore.score > highestScore:
    highestScore = eatScore.score
    actionToDo = Eat
  if sleepScore.score > highestScore:
    highestScore = sleepScore.score
    actionToDo = Sleep
  if playScore.score > highestScore:
    highestScore = playScore.score
    actionToDo = Play

  if decision.action != actionToDo:
    decision.action = actionToDo
    case actionToDo
    of Eat:
      game.world.rmComponent(entity, HasSleepAction)
      game.world.rmComponent(entity, HasPlayAction)
      game.world.mixEatAction(entity, 5'f32, 2'f32)
    of Sleep:
      game.world.rmComponent(entity, HasEatAction)
      game.world.rmComponent(entity, HasPlayAction)
      game.world.mixSleepAction(entity, 3'f32, 0.5'f32)
    of Play:
      game.world.rmComponent(entity, HasEatAction)
      game.world.rmComponent(entity, HasSleepAction)
      game.world.mixPlayAction(entity, 2'f32, 4'f32)

const Query = {HasHunger, HasDecision, HasTired, HasSleepScore, HasPlayScore}

proc sysActionScore =
  for entity, signature in game.world.signature.pairs:
    if signature * Query == Query:
      update(game, entity)

proc update(game: var Game, entity: Entity) =
  # Calculate scores
  template hunger: untyped = game.world.hunger[entity.idx]
  template decision: untyped = game.world.decision[entity.idx]

  if decision.action == Eat:
    # once it starts to eat, it will not stop until it's full
    eatScore.score = if hunger.value <= float32.epsilon: 0'f32 else: 1'f32
  else:
    let input = clamp(hunger.value * 0.01'f32, 0'f32, 1'f32)
    eatScore.score = exponentional(input, 2'f32)

  template tired: untyped = game.world.tired[entity.idx]
  template sleepScore: untyped = game.world.sleepScore[entity.idx]

  if decision.action == Sleep:
    # once it starts to sleep, it will not awake until it have enough rest
    sleepScore.score = if tired.value <= float32.epsilon: 0'f32 else: 1'f32
  else:
    let input = clamp(tired.value * 0.01'f32, 0'f32, 1'f32)
    sleepScore.score = raiseFastToSlow(input, 4)

  template playScore: untyped = game.world.playScore[entity.idx]

  # The play scorer has two considerations
  # The cat will play when it feels neigher hungry nor tired
  # Let's say it hate tired more(love to sleep), so the sleep consideration get more weight
  # sleep weight: 0.6, eat weight: 0.4

  let eatConcern = exponentional(clamp(hunger.value * 0.01'f32, 0'f32, 1'f32))
  let sleepConcern = raiseFastToSlow(clamp(tired.value * 0.01'f32, 0'f32, 1'f32))
  let concernBothersPlaying = sleepConcern * 0.6'f32 + eatConcern * 0.4'f32
  playScore.score = clamp(1'f32 - concernBothersPlaying, 0'f32, 1'f32)

proc sysAi(game: var Game) =
  const aiUpdateInterval = 4 - 1 # run every four engine updates
  if game.tickId and aiUpdateInterval == 0:
    sysActionScore(game)
    sysActionSelection(game)

const Query = {HasEatAction, HasHunger, HasTired}

proc sysEatAction =
  for entity, signature in game.world.signature.pairs:
    if signature * Query == Query:
      update(game, entity)

proc update(game: var Game, entity: Entity) =
  template tired: untyped = game.world.tired[entity.idx]
  template hunger: untyped = game.world.hunger[entity.idx]
  template eatAction: untyped = game.world.eatAction[entity.idx]

  # recover hungriness
  hunger.value = clamp(hunger.value - eatAction.hungerRecoverPerTick, 0'f32, 100'f32)
  # eat still get tired, but should slower than play
  tired.value = clamp(tired.value + eatAction.tirednessCostPerTick, 0'f32, 100'f32)

const Query = {HasSleepAction, HasHunger, HasTired}

proc sysSleepAction =
  for entity, signature in game.world.signature.pairs:
    if signature * Query == Query:
      update(game, entity)

proc update(game: var Game, entity: Entity) =
  template tired: untyped = game.world.tired[entity.idx]
  template hunger: untyped = game.world.hunger[entity.idx]
  template sleepAction: untyped = game.world.sleepAction[entity.idx]

  # recover tiredness
  tired.value = clamp(tired.value - sleepAction.tirednessRecoverPerTick, 0'f32, 100'f32)
  # sleep still get hungry slowly
  hunger.value = clamp(hunger.value + sleepAction.hungerCostPerTick, 0'f32, 100'f32)

const Query = {HasPlayAction, HasTired, HasHunger}

proc sysPlayAction =
  for entity, signature in game.world.signature.pairs:
    if signature * Query == Query:
      update(game, entity)

proc update(game: var Game, entity: Entity) =
  template tired: untyped = game.world.tired[entity.idx]
  template hunger: untyped = game.world.hunger[entity.idx]
  template playAction: untyped = game.world.playAction[entity.idx]

  hunger.value = clamp(hunger.value + playAction.hungerCostPerTick, 0'f32, 100'f32)
  tired.value = clamp(tired.value + playAction.tirednessCostPerTick, 0'f32, 100'f32)

proc update(game: var Game) =
  sysEatAction(game)
  sysSleepAction(game)
  sysPlayAction(game)
  sysAi(game)
  # Increment the Game engine tick
  inc(game.tickId)
