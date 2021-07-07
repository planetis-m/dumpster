
const
  cacheLineSize = 64

template Pad: untyped = (cacheLineSize - 1) div sizeof(T) + 1

type
  SpscQueue*[Cap: static[int]; T] = object
    data: array[Cap + 2 * Pad, T]

var
  s: SpscQueue[4, T = int]

echo s.data.len
