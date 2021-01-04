# http://blog.initprogram.com/2010/10/14/a-quick-basic-primer-on-the-irc-protocol/
# Original python code by gitgood
import random, net, strutils, strformat

const
   phrases = [
      "It is certain.",         # First ten phrases are "positive"
      "It is decidedly so.",
      "Without a doubt.",
      "Yes, definitely.",
      "You may rely on it.",
      "As I see it, yes.",
      "Most likely.",
      "Outlook good.",
      "Yes.",
      "Signs point to yes.",
      "Reply hazy, try again.", # Five "neutral" phrases.
      "Ask again later.",
      "Better not tell you now.",
      "Cannot predict now.",
      "Concentrate and ask again.",
      "Don't count on it.",     # Five "negative" phrases.
      "My reply is no.",
      "My sources say no.",
      "Outlook not so good.",
      "Very doubtful."]
   # Could be program arguments
   server = "irc.freenode.net"
   channel = "#8ballbottest"
   nickname = "magicbottest"
   command = ":!8ball"

proc magicbot() =
   # Create a socket instance.
   let irc = newSocket(Domain.AF_INET, SockType.SOCK_STREAM)
   irc.connect(server, Port(6667))
   echo(&"Connected to {server}.")
   irc.send(&"""USER {nickname} 0 * :Magicbot\r\n
                NICK {nickname}\r\n
                JOIN {channel}\r\n""")
   # Recieve data from the socket.
   var recieved = irc.recvLine()
   while recieved.len > 0:
      # Remove any trailing whitespace characters such as '\r' and '\n'
      recieved.stripLineEnd()
      #echo(recieved)
      # If the server sends a PING.
      if recieved.startsWith("PING"):
         # Respond with a PONG to prevent timing out.
         irc.send(&"PONG {recieved.substr(4)}\r\n")
         echo("Ponged")
      # If the message is directed to Magicbot
      elif (let found = recieved.find(command); found) > 0:
         # Get what's after "!8ball"
         let question = recieved.substr(found + command.len)
         # If there has been a statement after !8ball such as "!8ball am I going to die?"
         if isEmptyOrWhitespace(question):
            echo(&"The question was: {question}")
            # Then send a a random phrase from the phrases array to the channel.
            irc.send(&"PRIVMSG {channel} :{sample(phrases)}\r\n")
      recieved = irc.recvLine()
   # Recieved is empty string so the server disconnected.
   echo(&"{server} disconnected, exiting.")

magicbot()
