import random_forest, std/[parsecsv, strutils]

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

proc accuracy(X: seq[Features], yTrue: seq[int32]): float32 =
  var numCorrect = 0
  for i in 0..<IrisDataLen:
    if yTrue[i] == predict(X[i]).int32:
      inc numCorrect
  result = (numCorrect.float32/IrisDataLen)*100

proc main =
  let iris = readIrisData()
  echo "Accuracy: ", accuracy(iris.xs, iris.ys), "%"

main()
