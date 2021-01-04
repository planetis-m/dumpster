# https://docs.python.org/3/howto/sockets.html
# https://stackoverflow.com/questions/8627986/how-to-keep-a-socket-open-until-client-closes-it
# https://stackoverflow.com/questions/10091271/how-can-i-implement-a-simple-web-server-using-python-without-using-any-libraries
import net, strutils

proc createServer() =
  let serversocket = newSocket()
  try:
    serversocket.bindAddr(Port(9000), "localhost")
    serversocket.listen(5)
    while true:
      var
        clientsocket: Socket
        address: string
      serversocket.acceptAddr(clientsocket, address)

      var buffer = newString(5000)
      try:
        discard clientsocket.recv(buffer, buffer.len, 10000)
      except: discard
      echo buffer

      var data = "HTTP/1.1 200 OK\c\L"
      data.add "Content-Type: text/html; charset=utf-8\c\L"
      data.add "\c\L"
      data.add "<html><body>Hello World</body></html>\c\L\c\L"
      clientsocket.send(data)
      clientsocket.close()
  except:
    echo("Error: ", getCurrentExceptionMsg())

  serversocket.close()

echo("Access http://localhost:9000")
createServer()
