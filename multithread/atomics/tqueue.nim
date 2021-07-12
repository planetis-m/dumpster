# --threads:on --panics:on --gc:arc -d:useMalloc -t:"-O3 -fsanitize=thread" -l:"-fsanitize=thread" -d:danger -g
import queue

const
  seed = 99
  numIters = 1000

var
  q: MpscQueue[int]
  thr1, thr2: Thread[void]

proc producer =
  for i in 0 ..< numIters:
    q.push(i + seed)
    #echo " >> pushed ", i+seed

proc consumer =
  for x in q.items:
    echo " >> popped ", x

proc testMpScQueue =
  createThread(thr1, producer)
  producer()
  createThread(thr2, consumer)
  joinThread(thr1)
  joinThread(thr2)
  echo "main:"
  consumer()

testMpScQueue()
