import csp, tables, options

type
  MapColoringConstraint = Constraint[string, string]

proc newMapColoringConstraint(place1, place2: string): MapColoringConstraint =
  proc satisfied(assignment: Table[string, string]): bool =
    if place1 notin assignment or place2 notin assignment:
      result = true
    else:
      # check the color assigned to place1 is not the same as the
      # color assigned to place2
      result = assignment[place1] != assignment[place2]
  result = MapColoringConstraint(variables: @[place1, place2],
      satisfied: satisfied)

proc main =
  let variables = @["Western Australia", "Northern Territory", "South Australia",
                    "Queensland", "New South Wales", "Victoria", "Tasmania"]
  var domains: Table[string, seq[string]]
  for variable in variables.items:
    domains[variable] = @["red", "green", "blue"]
  var csp = initCsp(variables, domains)
  csp.addConstraint(newMapColoringConstraint("Western Australia",
      "Northern Territory"))
  csp.addConstraint(newMapColoringConstraint("Western Australia",
      "South Australia"))
  csp.addConstraint(newMapColoringConstraint("South Australia",
      "Northern Territory"))
  csp.addConstraint(newMapColoringConstraint("Queensland",
      "Northern Territory"))
  csp.addConstraint(newMapColoringConstraint("Queensland", "South Australia"))
  csp.addConstraint(newMapColoringConstraint("Queensland", "New South Wales"))
  csp.addConstraint(newMapColoringConstraint("New South Wales",
      "South Australia"))
  csp.addConstraint(newMapColoringConstraint("Victoria", "South Australia"))
  csp.addConstraint(newMapColoringConstraint("Victoria", "New South Wales"))
  csp.addConstraint(newMapColoringConstraint("Victoria", "Tasmania"))
  let solution = csp.backtrackingSearch()
  if solution.isNone:
    echo("No solution found!")
  else:
    echo(solution.get)

# {"Western Australia": "red", "South Australia": "blue", "Tasmania": "green",
#  "Northern Territory": "green", "Queensland": "red", "New South Wales": "green", "Victoria": "red"}

main()
