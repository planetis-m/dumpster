import std/[tables, options, sugar]

type
  Constraint*[V, D] = object
    # Base class for all constraints
    variables*: seq[V] # The variables that the constraint is between
    satisfied*: proc (assignment: Table[V, D]): bool

  Csp*[V, D] = object
    # A constraint satisfaction problem consists of variables of type V
    # that have ranges of values known as domains of type D and constraints
    # that determine whether a particular variable's domain selection is valid
    variables: seq[V] # variables to be constrained
    domains: Table[V, seq[D]] # domain of each variable
    constraints: Table[V, seq[Constraint[V, D]]]

proc initCsp*[V, D](variables: sink seq[V], domains: sink Table[V, seq[D]]): Csp[V, D] =
  result = Csp[V, D](variables: variables, domains: domains)
  for variable in result.variables.items:
    result.constraints[variable] = @[]
    if variable notin result.domains:
      raise newException(KeyError, "Every variable should have a domain assigned to it.")

proc addConstraint*[V, D](self: var Csp[V, D], constraint: sink Constraint[V, D]) =
  for variable in constraint.variables.items:
    if variable notin self.variables:
      raise newException(KeyError, "Variable in constraint not in CSP")
    else:
      self.constraints[variable].add(constraint)

proc isConsistent[V, D](self: Csp[V, D], variable: V, assignment: Table[V, D]): bool =
  # Check if the value assignment is consistent by checking all constraints
  # for the given variable against it
  for constraint in self.constraints[variable].items:
    if not constraint.satisfied(assignment):
      return false
  result = true

proc backtrackingSearch*[V, D](self: Csp[V, D], assignment: Table[V,
    D] = initTable[V, D]()): Option[Table[V, D]] =
  # assignment is complete if every variable is assigned (our base case)
  if len(assignment) == len(self.variables):
    return some(assignment)

  # get all variables in the CSP but not in the assignment
  let unassigned = collect(newSeq, for v in self.variables: (if v notin
      assignment: v))

  # get the every possible domain value of the first unassigned variable
  let first = unassigned[0]
  for value in self.domains[first].items:
    var localAssignment = assignment
    localAssignment[first] = value
    # if we're still consistent, we recurse (continue)
    if self.isConsistent(first, localAssignment):
      result = self.backtrackingSearch(localAssignment)
      # if we didn't find the result, we will end up backtracking
      if result.isSome():
        return
  return none(Table[V, D])

# proc hash*[V, D](a: Table[V, D]): Hash = hash($a)
#
# proc depthFirstSearch*[V, D](self: Csp[V, D]): Option[Table[V, D]] =
#    var frontier = newSeq[Table[V, D]]()
#    var discovered = initHashSet[Table[V, D]]()
#    frontier.add(initTable[V, D]())
#    discovered.incl(initTable[V, D]())
#    while frontier.len > 0:
#       let assignment = frontier.pop()
#       if len(assignment) == len(self.variables):
#          return some(assignment)
#       let unassigned = collect(newSeq, for v in self.variables: (if v notin assignment: v))
#       let first = unassigned[0]
#       for value in self.domains[first].items:
#          var localAssignment = assignment
#          localAssignment[first] = value
#          if self.isConsistent(first, localAssignment):
#             if localAssignment notin discovered:
#                frontier.add(localAssignment)
#                discovered.incl(localAssignment)
#    result = none(Table[V, D])
