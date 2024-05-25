import random_forest, std/[parsecsv, strutils, math]

const
  IrisDataLen = 150
  IrisFeatures = 4
  IrisLabels = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
  ]

type
  Features = array[IrisFeatures, float32]
  DataSet = object
    xs: seq[Features]
    ys: seq[int32]

proc readIrisData(): DataSet =
  var p: CsvParser
  try:
    p.open("iris.data")
    # p.readHeaderRow()
    var xs = newSeq[Features](IrisDataLen)
    var ys = newSeq[int32](IrisDataLen)
    var x = 0
    while p.readRow():
      for y in 0..<IrisFeatures:
        xs[x][y] = parseFloat(p.row[y])
      ys[x] = find(IrisLabels, p.row[^1]).int32
      inc x
    assert x == IrisDataLen
    result = DataSet(xs: xs, ys: ys)
  finally:
    p.close()

proc score(X: seq[Features], yTrue: seq[int32]): tuple[accuracy, precision, recall, f1: float32] =
  var tp, fp, tn, fn: array[IrisLabels.len, int]

  for i in 0..<IrisDataLen:
    let trueLabel = yTrue[i]
    let predLabel = predict(X[i]).int32

    if trueLabel == predLabel:
      inc tp[trueLabel]
    else:
      inc fp[predLabel]
      inc fn[trueLabel]
    for j in 0..<IrisLabels.len:
      if j != trueLabel and j != predLabel:
        inc tn[j]

  var accuracy: float32 = 0
  var precision, recall, f1: array[IrisLabels.len, float32]

  for i in 0..<IrisLabels.len:
    let tpVal = tp[i].float32
    let fpVal = fp[i].float32
    let fnVal = fn[i].float32
    let tnVal = tn[i].float32

    accuracy += (tpVal + tnVal)/(tpVal + fpVal + fnVal + tnVal)

    if tpVal + fpVal > 0:
      precision[i] = tpVal/(tpVal + fpVal)
    else:
      precision[i] = 0

    if tpVal + fnVal > 0:
      recall[i] = tpVal/(tpVal + fnVal)
    else:
      recall[i] = 0

    if precision[i] + recall[i] > 0:
      f1[i] = 2*(precision[i]*recall[i])/(precision[i] + recall[i])
    else:
      f1[i] = 0

  accuracy = accuracy/IrisLabels.len.float32
  let avgPrecision = sum(precision)/precision.len.float32
  let avgRecall = sum(recall)/recall.len.float32
  let avgF1 = sum(f1)/f1.len.float32

  result = (accuracy, avgPrecision, avgRecall, avgF1)

proc main =
  let iris = readIrisData()
  let scores = score(iris.xs, iris.ys)
  echo "Accuracy: ", scores.accuracy*100, "%"
  echo "Precision: ", scores.precision*100, "%"
  echo "Recall: ", scores.recall*100, "%"
  echo "F1-score: ", scores.f1*100, "%"

main()

# Scores on the Iris Dataset Extended:

# Random Forest:
# Accuracy: 95.611115%
# Precision: 93.43219%
# Recall: 93.416664%
# F1-score: 93.411865%

# Decision Tree:
# Accuracy: 95.388885%
# Precision: 93.555405%
# Recall: 93.083336%
# F1-score: 93.05284%
