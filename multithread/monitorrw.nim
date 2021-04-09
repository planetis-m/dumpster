import nlocks

# Reader-Writer problem using monitors
type
  RwMonitor* = object
    canRead: Cond # condition variable to check whether reader can read
    canWrite: Cond # condition variable to check whether writer can write
    condLock: Lock # mutex for synchronisation
    rcnt: int # number of readers
    wcnt: int # number of writers
    waitr: int # number of readers waiting
    waitw: int # number of writers waiting

proc initRwMonitor*(rw: var RwMonitor) =
  rw.rcnt = 0
  rw.wcnt = 0
  rw.waitr = 0
  rw.waitw = 0
  initCond(rw.canRead)
  initCond(rw.canWrite)
  initLock(rw.condLock)

proc destroyRwMonitor*(rw: var RwMonitor) {.inline.} =
  deinitCond(rw.canRead)
  deinitCond(rw.canWrite)
  deinitLock(rw.condLock)

proc beginRead*(rw: var RwMonitor) =
  # mutex provide synchronisation so that no other thread
  # can change the value of data
  acquire(rw.condLock)
  # if there are active or waiting writers
  if rw.wcnt == 1 or rw.waitw > 0:
    # incrementing waiting readers
    inc rw.waitr
    # reader suspended
    wait(rw.canRead, rw.condLock)
    dec rw.waitr
  # else reader reads the resource
  inc rw.rcnt
  release(rw.condLock)
  broadcast(rw.canRead)

proc endRead*(rw: var RwMonitor) =
  # if there are no readers left then writer enters monitor
  acquire(rw.condLock)
  dec rw.rcnt
  if rw.rcnt == 0:
    signal(rw.canWrite)
  release(rw.condLock)

proc beginWrite*(rw: var RwMonitor) =
  acquire(rw.condLock)
  # a writer can enter when there are no active
  # or waiting readers or other writer
  if rw.wcnt == 1 or rw.rcnt > 0:
    inc rw.waitw
    wait(rw.canWrite, rw.condLock)
    dec rw.waitw
  rw.wcnt = 1
  release(rw.condLock)

proc endWrite*(rw: var RwMonitor) =
  acquire(rw.condLock)
  rw.wcnt = 0
  # if any readers are waiting, threads are unblocked
  if rw.waitr > 0:
    signal(rw.canRead)
  else:
    signal(rw.canWrite)
  release(rw.condLock)
