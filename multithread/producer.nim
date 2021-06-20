import std/[locks, random, deques, os]

const
  max = 10

# Declaring global variables
var
  sum_B = 0
  sum_C = 0
  producerCount = 0
  consumerCount1 = 0
  consumerCount2 = 0
  Q: Deque[int] # Shared queue
  # Getting the mutex
  mutex: Lock
  dataNotProduced: Cond
  dataNotConsumed: Cond
  producerThread, consumerThread1, consumerThread2: Thread[void]

# Function to generate random numbers and
# push them into queue using thread A
proc producerFun() {.thread.} =
  while true:
    # Getting the lock on queue using mutex
    acquire(mutex)
    if Q.len < max and producerCount < max:
      # Getting the random number
      let num = rand(0..10)
      echo "Produced:  ", num
      # Pushing the number into queue
      Q.addLast(num)
      inc producerCount
      broadcast(dataNotProduced)
    # If queue is full, release the lock and return
    elif producerCount == max:
      release(mutex)
      return
    # If some other thread is exectuing, wait
    else:
      echo ">> Producer is in wait.."
      wait(dataNotConsumed, mutex)
    # Get the mutex unlocked
    release(mutex)
    sleep(200)

# Function definition for consumer thread B
proc addB() {.thread.} =
  while true:
    # Getting the lock on queue using mutex
    acquire(mutex)
    # Pop only when queue has at least 1 element
    if Q.len > 0:
      # Get the data from the front of queue
      let data = Q.popFirst()
      echo "B thread consumed: ", data
      # Add the data to the integer variable
      # associated with thread B
      sum_B += data
      inc consumerCount1
      signal(dataNotConsumed)
    # Check if consumed numbers from both threads
    # has reached to max value
    elif consumerCount2 + consumerCount1 == max:
      release(mutex)
      return
    # If some other thread is exectuing, wait
    else:
      echo "B is in wait.."
      wait(dataNotProduced, mutex)
    # Get the mutex unlocked
    release(mutex)
    sleep(200)

# Function definition for consumer thread C
proc addC() {.thread.} =
  while true:
    # Getting the lock on queue using mutex
    acquire(mutex)
    # Pop only when queue has at least 1 element
    if Q.len() > 0:
      # Get the data from the front of queue
      let data = Q.popFirst()
      echo "C thread consumed: ", data
      # Add the data to the integer variable
      # associated with thread B
      sum_C += data
      inc consumerCount2
      signal(dataNotConsumed)
    # Check if consmed numbers from both threads
    # has reached to max value
    elif consumerCount2 + consumerCount1 == max:
      release(mutex)
      return
    # If some other thread is exectuing, wait
    else:
      echo ">> C is in wait.."
      # Wait on a condition
      wait(dataNotProduced, mutex)
    # Get the mutex unlocked
    release(mutex)
    sleep(200)

# Driver code
proc main() =
  initCond(dataNotConsumed)
  initCond(dataNotProduced)
  initLock(mutex)
  # Initialising the seed
  randomize()
  # Function to create threads
  createThread(producerThread, producerFun)
  createThread(consumerThread1, add_B)
  createThread(consumerThread2, add_C)
  # join suspends execution of the calling
  # thread until the target thread terminates
  joinThread(producerThread)
  joinThread(consumerThread1)
  joinThread(consumerThread2)
  # Checking for the final value of thread
  if sum_C > sum_B:
    echo "Winner is  Thread C"
  elif sum_C < sum_B:
    echo "Winner is  Thread B"
  else:
    echo "Both has same score"
  deinitCond(dataNotConsumed)
  deinitCond(dataNotProduced)
  deinitLock(mutex)

main()
