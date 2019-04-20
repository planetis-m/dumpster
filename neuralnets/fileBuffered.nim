from strutils import Whitespace

const
   bufferSize = 8192

iterator buffered*(file: File; seps: set[char] = Whitespace): (int, TaintedString) {.
   tags: [ReadIOEffect].} =
   var
      buffer = TaintedString(newString(bufferSize))
      start = 0

   while true:
      let bytesRead = readBuffer(file, addr(buffer[start]), bufferSize - start)
      # Trim partially read content
      var chunkLen = start + bytesRead
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
