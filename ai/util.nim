# https://www.youtube.com/watch?v=M0Sx_M61ILU
import macros, random, strformat, fusion/astdsl

const
  maxHealingPotions = 3

type
  Player = object
    health: float32
    healingPotions: int

  Action = enum
    Attack, Heal, RunAway

proc initPlayer(): Player =
  result = Player(health: 1, healingPotions: maxHealingPotions)

# Considerations
proc healthWeight(p: Player): float32 =
  result = p.health

proc healingPotionsWeight(p: Player): float32 =
  result = p.healingPotions / maxHealingPotions

# Curve
proc boundedLinear(value: float32): float32 =
  result = max(1 - value, 0.5'f32)

proc inverse(value, factor, offset: float32): float32 =
  result = 1 / (value * factor + offset)

proc aboveZero(value: float32): float32 =
  result = if value > 0: 1 else: 0

proc equalsZero(value: float32): float32 =
  result = if value == 0: 1 else: 0

# Reasoner
proc attackWeight(enemy: Player): float32 =
  result = boundedLinear(enemy.healthWeight())
  echo &"Attack weight {result:.3}"

proc healWeight(p: Player): float32 =
  let healthWeight = inverse(p.healthWeight(), 10, 1)
  let healingPotionsWeight = aboveZero(p.healingPotionsWeight())
  result = healthWeight * healingPotionsWeight
  echo &"Health weight {healthWeight:.3} Healing Potions weight {healingPotionsWeight:.3} Heal weight {result:.3}"

proc runAwayWeight(p: Player): float32 =
  let healthWeight = inverse(p.healthWeight(), 10, 1)
  let healingPotionsWeight = equalsZero(p.healingPotionsWeight())
  result = healthWeight * healingPotionsWeight
  echo &"Health weight {healthWeight:.3} Healing Potions weight {healingPotionsWeight:.3} Run Away weight {result:.3}"

# Chooser
macro maxScore(args: varargs[untyped]): untyped =
  template action(x: NimNode): untyped = x[1][0]
  template score(x: NimNode): untyped = x[0]

  result = buildAst(stmtListExpr):
    let actionSym = genSym(nskVar, "action")
    let maxScoreSym = genSym(nskVar, "maxScore")
    expectMinLen(args, 1)
    newVarStmt(actionSym, args[0].action)
    newVarStmt(maxScoreSym, args[0].score)
    for i in 1..<args.len:
      ifStmt:
        elifBranch(infix(ident">", args[i].score, maxScoreSym)):
          stmtList:
            asgn(actionSym, args[i].action)
            asgn(maxScoreSym, args[i].score)
    actionSym

proc chooseAction(p, enemy: Player): Action =
  result = maxScore:
  of attackWeight(enemy): Attack
  of healWeight(p): Heal
  of runAwayWeight(p): RunAway

proc main =
  randomize()
  var players = @[initPlayer(), initPlayer()]
  while true:
    echo &"Player 1: health: {players[0].health:.3} potions {players[0].healingPotions}"
    let action1 = chooseAction(players[0], players[1])
    echo "Player 1 action: ", action1
    case action1
    of Attack:
      players[1].health -= rand(0.2'f32..0.3'f32)
      if players[1].health <= 0:
        echo "Player 1 wins!"
        break
    of Heal:
      dec players[0].healingPotions
      players[0].health += rand(0.2'f32..0.3'f32)
    of RunAway:
      echo "Player 1 ran away."
      break
    # Player 2
    echo &"Player 2: health: {players[1].health:.3} potions {players[1].healingPotions}"
    let action2 = chooseAction(players[1], players[0])
    echo "Player 2 action: ", action2
    case action2
    of Attack:
      players[0].health -= rand(0.2'f32..0.3'f32)
      if players[0].health <= 0:
        echo "Player 2 wins!"
        break
    of Heal:
      dec players[1].healingPotions
      players[1].health += rand(0.2'f32..0.3'f32)
    of RunAway:
      echo "Player 2 ran away."
      break

main()
