import strutils, parseutils

const
   bufferSize = 1024
   path = "saved.tinn"

var
   nweights = 1000
   weights = newSeq[float](nweights)

var
   file = open(path)
   buffer = newString(bufferSize)
   index = 0
   start = 0

while true:
   let bytesRead = readBuffer(file, addr(buffer[start]), bufferSize - start)
   # Trim partially read content
   var chunkLen = bytesRead
   while chunkLen >= 0 and buffer[chunkLen - 1] notin Whitespace:
      chunkLen.dec
   # Fill the weights array
   var chunkPos = 0
   while chunkPos < chunkLen:
      let bytesParsed = buffer.parseFloat(weights[index], chunkPos)
      assert bytesParsed > 0 and bytesParsed + chunkPos < chunkLen
      chunkPos += bytesParsed
      # Skip whitespace chars
      while chunkPos < chunkLen and buffer[chunkPos] in Whitespace:
         chunkPos.inc
      index.inc
   # Break iff buffer is only half-full
   if bytesRead < bufferSize - start:
      break
   # shift trimmed chars to the front
   var trimmedStart = chunkLen
   var trimmedLen = bufferSize - chunkLen
   start = 0
   while start < trimmedLen:
      buffer[start] = buffer[trimmedStart]
      trimmedStart.inc
      start.inc

iterator buffered(file: File; seps: set[char] = Whitespace): (int, TaintedString) {.
   tags: [ReadIOEffect].} =
   var
      buffer = TaintedString(newString(bufferSize))
      start = 0

   while true:
      let bytesRead = readBuffer(file, addr(buffer[start]), bufferSize - start)
      # Trim partially read content
      var chunkLen = bytesRead
      while chunkLen >= 0 and buffer[chunkLen - 1] notin seps:
         chunkLen.dec
      # Yield the buffer
      yield (chunkLen, buffer)
      # Break iff buffer is only half-full
      if bytesRead < bufferSize - start:
         break
      # shift trimmed chars to the front
      var trimmedStart = chunkLen
      var trimmedLen = bufferSize - chunkLen
      start = 0
      while start < trimmedLen:
         buffer[start] = buffer[trimmedStart]
         trimmedStart.inc
         start.inc

index = 0
for len, chunk in file.buffered:
   # Fill the weights array
   var chunkPos = 0
   while chunkPos < len:
      let bytesParsed = chunk.parseFloat(weights[index], chunkPos)
      assert bytesParsed > 0
      chunkPos += bytesParsed
      # Skip whitespace chars
      while chunkPos < len and chunk[chunkPos] in Whitespace:
         chunkPos.inc
      index.inc
