import jsonpak, jsonpak/[builder, mapper, jsonptr, extra], std/macros

type
  TargetNames = enum
    setosa, versicolor, virginica

  FeatureNames = enum
    sepal_length, sepal_width, petal_length, petal_width

const
  classifier = %*{
    "node": {
      "feature": petal_length,
      "threshold": 2.45,
      "left": {
        "class": setosa
      },
      "right": {
        "feature": petal_width,
        "threshold": 1.75,
        "left": {
          "feature": petal_length,
          "threshold": 4.95,
          "left": {
            "feature": petal_width,
            "threshold": 1.65,
            "left": {
              "class": versicolor
            },
            "right": {
              "class": virginica
            }
          },
          "right": {
            "class": virginica
          }
        },
        "right": {
          "class": virginica
        }
      }
    }
  }

proc generateNode(nodeData: JsonTree, input: NimNode): NimNode =
  let res = ident"result"
  if nodeData.contains(JsonPtr"/class"):
    result = newAssignment(res, newLit(fromJson(nodeData, JsonPtr"/class", TargetNames)))
  else:
    result = newNimNode(nnkIfStmt)
    result.add(newTree(nnkElifBranch,
        infix(newTree(nnkBracketExpr, input, newLit(fromJson(nodeData, JsonPtr"/feature", FeatureNames).int)),
        "<=", newLit(fromJson(nodeData, JsonPtr"/threshold", float32))),
        generateNode(extract(nodeData, JsonPtr"/left"), input)))
    result.add(newTree(nnkElse, generateNode(extract(nodeData, JsonPtr"/right"), input)))

macro generateClassifier(classifierData: static JsonTree): untyped =
  let input = genSym(nskParam, "input")
  result = newProc(name = ident"predict",
      params = [bindSym"TargetNames", newIdentDefs(name = input,
      kind = newTree(nnkBracketExpr, bindSym"seq", bindSym"float32"))],
      body = generateNode(extract(classifierData, JsonPtr"/node"), input))
  echo result.repr

generateClassifier(classifier)

echo predict(@[4.9'f32,3.1,1.5,0.1]) # == setosa
