import std/setutils, jsonpak, jsonpak/[mapper, jsonptr], jsonpak/private/[jsonnode, jsontree]

const
  classifier = %*{
    "node": {
      "feature": "petal_length",
      "threshold": 2.45,
      "left": {
        "class": "setosa"
      },
      "right": {
        "feature": "petal_width",
        "threshold": 1.75,
        "left": {
          "feature": "petal_length",
          "threshold": 4.95,
          "left": {
            "feature": "petal_width",
            "threshold": 1.65,
            "left": {
              "class": "versicolor"
            },
            "right": {
              "class": "virginica"
            }
          },
          "right": {
            "class": "virginica"
          }
        },
        "right": {
          "class": "virginica"
        }
      }
    }
  }

type
  ValidationErrorKind* = enum
    InvalidType = "Invalid type"
    MissingRequiredProperty = "Missing required property"
    AdditionalPropertiesNotAllowed = "Additional properties not allowed"
    DuplicateKey = "Duplicate key found"
    MinLengthNotSatisfied = "String length is less than minLength"
    MaxLengthExceeded = "String length exceeds maxLength"
    PatternNotMatched = "String does not match the pattern"
    MinItemsNotSatisfied = "Array has fewer items than minItems"
    MaxItemsExceeded = "Array has more items than maxItems"
    UniqueItemsNotSatisfied = "Array contains duplicate items"
    MinimumNotSatisfied = "Number is less than the minimum"
    MaximumExceeded = "Number exceeds the maximum"
    MultipleOfNotSatisfied = "Number is not a multiple of the specified value"
    EnumValueNotAllowed = "Value is not allowed by the enum"
    InvalidFormat = "Invalid format"
    OneOfNotSatisfied = "Value does not satisfy any of the subschemas in oneOf"
    AllOfNotSatisfied = "Value does not satisfy all the subschemas in allOf"
    AnyOfNotSatisfied = "Value does not satisfy any of the subschemas in anyOf"
    NotSchemaNotSatisfied = "Value matches the schema in not"
    DependenciesNotSatisfied = "Dependencies are not satisfied"
    InvalidReference = "Invalid ref value"
    InvalidSchema = "Invalid JSON schema"

  ValidationError* = object
    kind*: ValidationErrorKind
    path*: JsonPtr

proc `$`*(error: ValidationError): string =
  result = $error.kind & " (path: " & error.path.string & ')'

type
  IrisClassifierSchema = object

template adderr(a: ValidationErrorKind, b: string) =
  errors.add(ValidationError(kind: a, path: b.JsonPtr))
  result = false

proc isValidNodeTuple(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  result = true
  if n.kind == opcodeArray:
    var i = 0
    let oldLen = path.len
    for x in sonsReadonly(tree, n):
      path.add '/'
      path.addInt i
      case i
      of 0:
        if x.kind != opcodeString:
          adderr(InvalidType, path)
        if x.str notin ["setosa", "versicolor", "virginica"]:
          adderr(EnumValueNotAllowed, path)
      of 1:
        if x.kind == opcodeInt or x.kind == opcodeFloat:
          adderr(InvalidType, path)
      else: discard
      path.setLen(oldLen)
      inc i
    if i > 1:
      adderr(MaxItemsExceeded, path)
    if i < 1:
      adderr(MinItemsNotSatisfied, path)
  else:
    adderr(InvalidType, path)

proc isValidNodeArray(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  result = true
  if n.kind == opcodeArray:
    var i = 0
    let oldLen = path.len
    for x in sonsReadonly(tree, n):
      path.add '/'
      path.addInt i
      if x.kind != opcodeString:
        adderr(InvalidType, path)
      if x.str notin ["setosa", "versicolor", "virginica"]:
        adderr(EnumValueNotAllowed, path)
      path.setLen(oldLen)
      inc i
    if i > 3:
      adderr(MaxItemsExceeded, path)
    if i < 0:
      adderr(MinItemsNotSatisfied, path)
  else:
    adderr(InvalidType, path)

proc isValidNodeOneOf(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool

proc isValidNodeBranch1(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  type
    NodeKey = enum
      class
  const optionalKeys: set[NodeKey] = {}
  result = true
  if n.kind == opcodeObject:
    var nodeKeys: set[NodeKey] = {}
    let oldLen = path.len
    for x in keys(tree, n):
      path.add '/'
      path.add x.str
      case x.str
      of "class":
        if NodeKey.class in nodeKeys:
          adderr(DuplicateKey, path)
        nodeKeys.incl NodeKey.class
        let classVal = x.firstSon
        if classVal.kind != opcodeString:
          adderr(InvalidType, path)
        if classVal.str notin ["setosa", "versicolor", "virginica"]:
          adderr(EnumValueNotAllowed, path)
      else:
        adderr(AdditionalPropertiesNotAllowed, path)
      path.setLen(oldLen)
    if not (fullSet(NodeKey) - optionalKeys <= nodeKeys):
      for e in low(NodeKey)..high(NodeKey):
        if e notin optionalKeys+nodeKeys:
          adderr(MissingRequiredProperty, path & '/' & $e)
  else:
    adderr(InvalidType, path)

proc isValidNodeBranch2(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  type
    NodeKey = enum
      feature, threshold, left, right
  const optionalKeys: set[NodeKey] = {left, right}
  result = true
  if n.kind == opcodeObject:
    var nodeKeys: set[NodeKey] = {}
    let oldLen = path.len
    for x in keys(tree, n):
      path.add '/'
      path.add x.str
      case x.str
      of "feature":
        if NodeKey.feature in nodeKeys:
          adderr(DuplicateKey, path)
        nodeKeys.incl NodeKey.feature
        let featureVal = x.firstSon
        if featureVal.kind != opcodeString:
          adderr(InvalidType, path)
        if featureVal.str notin ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
          adderr(EnumValueNotAllowed, path)
      of "threshold":
        if NodeKey.threshold in nodeKeys:
          adderr(DuplicateKey, path)
        nodeKeys.incl NodeKey.threshold
        let thresholdVal = x.firstSon
        if thresholdVal.kind != opcodeFloat:
          adderr(InvalidType, path)
      of "left":
        if NodeKey.left in nodeKeys:
          adderr(DuplicateKey, path)
        nodeKeys.incl NodeKey.left
        result = result and isValidNodeOneOf(tree, x.firstSon, path, errors)
      of "right":
        if NodeKey.right in nodeKeys:
          adderr(DuplicateKey, path)
        nodeKeys.incl NodeKey.right
        result = result and isValidNodeOneOf(tree, x.firstSon, path, errors)
      else:
        adderr(AdditionalPropertiesNotAllowed, path)
      path.setLen(oldLen)
    if not (fullSet(NodeKey) - optionalKeys <= nodeKeys):
      for e in low(NodeKey)..high(NodeKey):
        if e notin nodeKeys+optionalKeys:
          adderr(MissingRequiredProperty, path & '/' & $e)
  else:
    adderr(InvalidType, path)

# anyOf
proc isValidNodeAnyOf(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  let oldErrLen = errors.len
  result = false
  if isValidNodeBranch1(tree, n, path, errors) or
      isValidNodeBranch2(tree, n, path, errors):
    result = true
    errors.setLen(oldErrLen)
  if not result:
    adderr(AnyOfNotSatisfied, path)

# oneOf
proc isValidNodeOneOf(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  let oldErrLen = errors.len
  result = false
  if isValidNodeBranch1(tree, n, path, errors) xor
      isValidNodeBranch2(tree, n, path, errors):
    result = true
    errors.setLen(oldErrLen)
  if not result:
    adderr(OneOfNotSatisfied, path)

# allOf
proc isValidNodeAllOf(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  let oldErrLen = errors.len
  result = false
  if isValidNodeBranch1(tree, n, path, errors) and
      isValidNodeBranch2(tree, n, path, errors):
    result = true
    errors.setLen(oldErrLen)
  if not result:
    adderr(AllOfNotSatisfied, path)

# not
proc isValidNodeNot(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  let oldErrLen = errors.len
  result = false
  if not isValidNodeBranch1(tree, n, path, errors) and
      not isValidNodeBranch2(tree, n, path, errors):
    result = true
    errors.setLen(oldErrLen)
  if not result:
    adderr(NotSchemaNotSatisfied, path)

proc isValidRoot(tree: JsonTree; n: NodePos; path: var string; errors: var seq[ValidationError]): bool =
  type
    NodeKey = enum
      node
  const optionalKeys: set[NodeKey] = {}
  result = true
  if n.kind == opcodeObject:
    let oldLen = path.len
    var nodeKeys: set[NodeKey] = {}
    for x in keys(tree, n):
      path.add '/'
      path.add x.str
      case x.str
      of "node":
        if NodeKey.node in nodeKeys:
          adderr(DuplicateKey, path)
        nodeKeys.incl NodeKey.node
        result = result and isValidNodeOneOf(tree, x.firstSon, path, errors)
      else:
        adderr(AdditionalPropertiesNotAllowed, path)
      path.setLen(oldLen)
    if not (fullSet(NodeKey) - optionalKeys <= nodeKeys):
      for e in low(NodeKey)..high(NodeKey):
        if e notin nodeKeys+optionalKeys:
          adderr(MissingRequiredProperty, path & '/' & $e)
  else:
    adderr(InvalidType, path)

proc isValid(tree: JsonTree; t: typedesc[IrisClassifierSchema]; errors: var seq[ValidationError]): bool =
  var path = ""
  result = isValidRoot(tree, rootNodeId, path, errors)

var errors: seq[ValidationError] = @[]
echo isValid(classifier, IrisClassifierSchema, errors)
if errors.len > 0:
  echo "Validation errors:"
  for err in errors:
    echo err
else:
  echo "Validation successful"
