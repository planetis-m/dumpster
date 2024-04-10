import jsonpak, jsonpak/[builder, mapper, parser, jsonptr, extra], std/[macros, sequtils]

type
  TargetNames = enum
    setosa, versicolor, virginica

const classifier = parseJson(readFile"rforest_min.json")

proc generateNode(nodeData: JsonTree, input: NimNode): NimNode =
  let res = ident"result"
  if nodeData.contains(JsonPtr"/class"):
    result = newAssignment(res, newLit(fromJson(nodeData, JsonPtr"/class", TargetNames)))
  else:
    result = newNimNode(nnkIfStmt)
    result.add(newTree(nnkElifBranch,
        infix(newTree(nnkBracketExpr, input, newLit(fromJson(nodeData, JsonPtr"/feature", int))),
        "<=", newLit(fromJson(nodeData, JsonPtr"/threshold", float32))),
        generateNode(extract(nodeData, JsonPtr"/left"), input)))
    result.add(newTree(nnkElse, generateNode(extract(nodeData, JsonPtr"/right"), input)))

macro generateClassifier(classifierData: static JsonTree): untyped =
  result = newStmtList()
  let ensemble = newStmtList()
  let counter = genSym(nskVar, "count")
  let input = genSym(nskParam, "input")
  for treeData in items(classifierData, JsonPtr"", JsonTree):
    let input2 = genSym(nskParam, "input")
    let estimator = genSym(nskProc, "predict")
    result.add newProc(name = estimator,
        params = [bindSym"TargetNames", newIdentDefs(name = input2,
        kind = newTree(nnkBracketExpr, bindSym"seq", bindSym"float32"))],
        body = generateNode(extract(treeData, JsonPtr"/node"), input2))
    ensemble.add newCall(bindSym"inc", newTree(nnkBracketExpr, counter, newCall(estimator, input)))
  result.add newProc(name = ident"predict", params = [bindSym"TargetNames",
      newIdentDefs(name = input, kind = newTree(nnkBracketExpr, bindSym"seq", bindSym"float32"))],
      body = newStmtList(newTree(nnkVarSection,
        newIdentDefs(counter, newTree(nnkBracketExpr, bindSym"array", bindSym"TargetNames", bindSym"int32"))),
        ensemble,
        newAssignment(ident"result", newCall(bindSym"TargetNames", newCall(bindSym"maxIndex", counter)))))
  echo result.repr

generateClassifier(classifier)

echo predict(@[4.9'f32,3.1,1.5,0.1]) # == setosa
