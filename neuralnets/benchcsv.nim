import parsecsv, streams

const
   path = "iris.data"
   ncol = 5

proc main =
   var s = newFileStream(path, fmRead)
   var x: CsvParser
   open(x, s, path)
   while readRow(x):
      echo x.row

   close(x)

main()
