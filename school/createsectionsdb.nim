import db_sqlite, strutils

const data = [
   ("IPE80",   80,  46, 3.8, 5.2,  5,  7.64,  6.0),
   ("IPE100", 100,  55, 4.1, 5.7,  7, 10.32,  8.1),
   ("IPE120", 120,  64, 4.4, 6.3,  7, 13.21, 10.4),
   ("IPE140", 140,  73, 4.7, 6.9,  7, 16.43, 12.9),
   ("IPE160", 160,  82, 5.0, 7.4,  9, 20.09, 15.8),
   ("IPE180", 180,  91, 5.3, 8.0,  9, 23.95, 18.8),
   ("IPE200", 200, 100, 5.6, 8.5, 12, 28.48, 22.4)]

let db = open(connection="sections.db", user="tony", password="",
              database="sections")
let model = readFile("sections_model.sql")
for m in model.split(';'):
   if m.strip != "":
      db.exec(sql(m), [])

# Insert data
for k, (name, h, b, tw, tf, r, area, weight) in data.pairs:
   db.exec(sql"insert into ipe(id, name, h, b, tw, tf, r, area, weight) values(?, ?, ?, ?, ?, ?, ?, ?, ?)",
           k + 1, name, h, b, tw, tf, r, area, weight)

db.close()
