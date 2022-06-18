# http://goran.krampe.se/2014/10/25/nim-socketserver/
import threadpool, net, os, selectors, strutils

##   nim c --threads:on --d:release spawningserver.nim
##
## Run it and throw wrk on it, 100 concurrent clients:
##   wrk -r -n 500000 -c 100 http://localhost:8099/
##

type
  Server = ref object
    socket: Socket

# Amount of data to send
const bytes = 100
# The payload
const content = repeat("x", bytes)
# And the response
const response = "HTTP/1.1 200 OK\r\LContent-Length: " & $content.len & "\r\L\r\L" & content

proc handle(client: Socket) =
  var buf = ""
  try:
    client.readLine(buf, timeout = 20000)
    client.send(response)
  finally:
    client.close()

proc loop(self: Server) =
  let selector = newSelector[int]()
  selector.registerHandle(self.socket.getFD, {Event.Read}, 0)
  while true:
    if selector.select(1000).len > 0:
      var client = Socket()
      accept(self.socket, client)
      spawn handle(client)

proc listen(self: Server, port: int) =
  self.socket = newSocket()
  try:
    self.socket.bindAddr(port = Port(port))
    self.socket.listen()
    echo("Server listening on port " & $port)
    self.loop()
  finally:
    self.socket.close()

when isMainModule:
  var port = 8099
  var server = Server()
  server.listen(port)
