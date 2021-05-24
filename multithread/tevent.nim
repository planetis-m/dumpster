import event, std/os

var
  thread: Thread[void]
  fuel = 0
  refueled: Event

proc refuel =
  fuel += 30
  sleep(1)
  signal(refueled)

proc main =
  initEvent refueled
  createThread(thread, refuel)
  wait refueled
  assert fuel == 30

  joinThread(thread)

main()
