import jsonpak, jsonpak/[builder, parser, jsonptr, extra], std/[macros, sequtils]

type
  TargetNames = enum
    setosa, versicolor, virginica

const classifier = parseJson(readFile"rforest_min.json")

proc generateNode(nodeData: JsonTree, input: NimNode): NimNode =
  if nodeData.contains(JsonPtr"/class"):
    # result = TargetNames(class)
    result = newAssignment(ident"result", newCall(bindSym"TargetNames",
        newLit(fromJson(nodeData, JsonPtr"/class", int))))
  else:
    result = newNimNode(nnkIfStmt)
    # if input[feature] <= threshold'f32:
    result.add(newTree(nnkElifBranch,
        infix(newTree(nnkBracketExpr, input, newLit(fromJson(nodeData, JsonPtr"/feature", int))),
        "<=", newLit(fromJson(nodeData, JsonPtr"/threshold", float32))),
        generateNode(extract(nodeData, JsonPtr"/left"), input)))
    # else:
    result.add(newTree(nnkElse, generateNode(extract(nodeData, JsonPtr"/right"), input)))

macro generateClassifier(classifierData: static JsonTree): untyped =
  result = newStmtList()
  let voting = newStmtList()
  let counter = genSym(nskVar, "count")
  let input = genSym(nskParam, "input")
  for treeData in items(classifierData, JsonPtr"", JsonTree):
    let input2 = genSym(nskParam, "input")
    let estimator = genSym(nskProc, "predict")
    # proc predict(input: openarray[float32]): TargetNames =
    result.add newProc(name = estimator,
        params = [bindSym"TargetNames", newIdentDefs(name = input2,
        kind = newTree(nnkBracketExpr, bindSym"openarray", bindSym"float32"))],
        body = generateNode(extract(treeData, JsonPtr"/node"), input2))
    # inc(count[predict(input)])
    voting.add newCall(bindSym"inc", newTree(nnkBracketExpr, counter, newCall(estimator, input)))
  # proc predict(input: openarray[float32]): TargetNames =
  result.add newProc(name = ident"predict", params = [bindSym"TargetNames",
      newIdentDefs(name = input, kind = newTree(nnkBracketExpr, bindSym"openarray", bindSym"float32"))],
      body = newStmtList(newTree(nnkVarSection,
        # var count: array[TargetNames, int32]
        newIdentDefs(counter, newTree(nnkBracketExpr, bindSym"array", bindSym"TargetNames", bindSym"int32"))),
        voting,
        # result = TargetNames(maxIndex(count))
        newAssignment(ident"result", newCall(bindSym"TargetNames", newCall(bindSym"maxIndex", counter)))))
  echo result.repr

generateClassifier(classifier)

# echo predict(@[4.9'f32,3.1,1.5,0.1]) # == setosa
# echo predict(@[6.7'f32,3.0,5.0,1.7]) # == versicolor
# echo predict(@[7.7'f32,2.6,6.9,2.3]) # == virginica
