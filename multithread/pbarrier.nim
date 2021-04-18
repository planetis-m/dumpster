import os

type
  PthreadAttr* {.byref, importc: "pthread_attr_t", header: "<sys/types.h>".} = object
  PthreadBarrier* {.byref, importc: "pthread_barrier_t", header: "<sys/types.h>".} = object

  Errno* = cint

var PTHREAD_BARRIER_SERIAL_THREAD* {.importc, header:"<pthread.h>".}: Errno

proc pthread_barrier_init*(
      barrier: PthreadBarrier,
      attr: PthreadAttr or ptr PthreadAttr,
      count: range[0'i32..high(int32)]
    ): Errno {.header: "<pthread.h>".}
  ## Initialize `barrier` with the attributes `attr`.
  ## The barrier is opened when `count` waiters arrived.

proc pthread_barrier_destroy*(
      barrier: sink PthreadBarrier): Errno {.header: "<pthread.h>".}
  ## Destroy a previously dynamically initialized `barrier`.

proc pthread_barrier_wait*(
      barrier: var PthreadBarrier
    ): Errno {.header: "<pthread.h>".}
  ## Wait on `barrier`
  ## Returns PTHREAD_BARRIER_SERIAL_THREAD for a single arbitrary thread
  ## Returns 0 for the other
  ## Returns Errno if there is an error

type Barrier* = PthreadBarrier

proc initBarrier*(syncBarrier: var Barrier, threadCount: range[0'i32..high(int32)]) {.inline.} =
  ## Initialize a synchronization barrier that will block ``threadCount`` threads
  ## before release.
  let err {.used.} = pthread_barrier_init(syncBarrier, nil, threadCount)
  when compileOption("assertions"):
    if err != 0:
      raiseOSError(OSErrorCode(err))

proc wait*(syncBarrier: var Barrier): bool {.inline, discardable.} =
  ## Blocks thread at a synchronization barrier.
  ## Returns true for one of the threads (the last one on Windows, undefined on Posix)
  ## and false for the others.
  let err {.used.} = pthread_barrier_wait(syncBarrier)
  when compileOption("assertions"):
    if err != PTHREAD_BARRIER_SERIAL_THREAD and err < 0:
      raiseOSError(OSErrorCode(err))
  result = if err == PTHREAD_BARRIER_SERIAL_THREAD: true
           else: false

proc destroyBarrier*(syncBarrier: sink Barrier) {.inline.} =
  ## Deletes a synchronization barrier.
  ## This assumes no race between waiting at a barrier and deleting it,
  ## and reuse of the barrier requires initialization.
  let err {.used.} = pthread_barrier_destroy(syncBarrier)
  when compileOption("assertions"):
    if err < 0:
      raiseOSError(OSErrorCode(err))
