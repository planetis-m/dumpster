
import asyncnet, asyncdispatch, tables

# marking it as threadvar to avoid GC-safety warnings/errors,
# we have 1 thread anyway
var clients {.threadvar.}: TableRef[int, AsyncSocket]
clients = newTable[int, AsyncSocket]()

proc processClient(id: int, client: AsyncSocket) {.async.} =
  while true:
    let line = await client.recvLine()
    if line == "":
      clients.del(id)
      break
    for c in clients.values():
      await c.send(line & "\c\L")

proc serve() {.async.} =
  var clientCtr = 0
  var server = newAsyncSocket()
  server.bindAddr(Port(12321))
  server.listen()

  while true:
    let client = await server.accept()
    clients.add(clientCtr, client)
    await processClient(clientCtr, client)
    inc clientCtr

asyncCheck serve()
runForever()
