type
  FormulaKind = enum
    fkVar,        ## element is a variable like 'X'
    fkLit,        ## element is a literal like 0.1
    fkAdd,        ## element is an addition operation
    fkMul,        ## element is a multiplication operation
    fkExp         ## element is an exponentiation operation

type
  Formula = ref object
    case kind: FormulaKind
    of fkVar: name: string
    of fkLit: value: float
    of fkAdd, fkMul, fkExp: left, right: Formula

from math import pow

proc evaluate(n: Formula, varToVal: proc (name: string): float): float =
  case n.kind
  of fkVar: varToVal(n.name)
  of fkLit: n.value
  of fkAdd: evaluate(n.left, varToVal) + evaluate(n.right, varToVal)
  of fkMul: evaluate(n.left, varToVal) * evaluate(n.right, varToVal)
  of fkExp: pow(evaluate(n.left, varToVal), evaluate(n.right, varToVal))

echo evaluate(Formula(kind: fkLit, value: 0.4), nil)

proc isPolyTerm(n: Formula): bool =
  n.kind == fkMul and n.left.kind == fkLit and (let e = n.right;
    e.kind == fkExp and e.left.kind == fkVar and e.right.kind == fkLit)

proc isPolynomial(n: Formula): bool =
  isPolyTerm(n) or
    (n.kind == fkAdd and isPolynomial(n.left) and isPolynomial(n.right))

let myFormula = Formula(kind: fkMul,
                        left: Formula(kind: fkLit, value: 2.0),
                        right: Formula(kind: fkExp,
                          left: Formula(kind: fkVar, name: "x"),
                          right: Formula(kind: fkLit, value: 5.0)))

echo isPolyTerm(myFormula)

proc pat2kind(pattern: string): FormulaKind =
  case pattern
  of "^": fkExp
  of "*": fkMul
  of "+": fkAdd
  of "x": fkVar
  of "c": fkLit
  else:   fkVar # no error reporting for reasons of simplicity

import macros

proc matchAgainst(n, pattern: NimNode): NimNode {.compileTime.} =
  template `@`(current, field: untyped): untyped =
    newDotExpr(current, newIdentNode(astToStr(field)))

  template `==@`(n, pattern: untyped): untyped =
    newCall("==", n@kind, newIdentNode($pat2kind(pattern.strVal)))

  case pattern.kind
  of CallNodes:
    result = newCall("and",
      n ==@ pattern[0],
      matchAgainst(n@left, pattern[1]))
    if pattern.len == 3:
      result = newCall("and", result.copy,
        matchAgainst(n@right, pattern[2]))
  of nnkIdent:
    result = n ==@ pattern
  of nnkPar:
    result = matchAgainst(n, pattern[0])
  else:
    error "invalid pattern"

macro `=~` (n: Formula, pattern: untyped): bool =
  result = matchAgainst(n, pattern)

proc isPolyTerm2(n: Formula): bool = n =~ c * x^c

echo isPolyTerm2(myFormula), isPolyTerm2(Formula(kind: fkLit, value: 0.7))

import std/tables

let lut = {
  "x": 1.0
}.toTable

echo evaluate(myFormula, proc (name: string): float = lut.getOrDefault(name))
