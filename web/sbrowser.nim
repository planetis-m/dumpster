import net

let client = newSocket()
client.connect("data.pr4e.org", Port(80))
client.send("GET http://data.pr4e.org/page1.htm HTTP/1.0\c\L\c\L")

var buffer = newString(512)
while client.recv(buffer, buffer.len) > 0:
  stdout.write buffer

client.close()
