import macros, math

when defined(NeuralFloat32):
   type NeuralFloat* = float32
else:
   type NeuralFloat* = float64

type
   ActivationFunction* = object
      fn*, deriv*: proc(x: NeuralFloat): NeuralFloat {.nimcall.}

macro act(body: untyped): untyped =
   result = nnkObjConstr.newTree(ident"ActivationFunction")
   let flType = ident"NeuralFloat"
   expectLen(body, 2)
   for fn in body:
      expectKind(fn, nnkCall)
      expectLen(fn, 3)
      let fnParam = newIdentDefs(fn[1], flType)
      let p = newProc(newEmptyNode(), [flType, fnParam], fn[2], nnkLambda)
      result.add nnkExprColonExpr.newTree(fn[0], p)

let
   identity* = act:
      fn(x): x
      deriv(fx): 1
   sigmoid* = act:
      fn(x): 1 / (1 + exp(-x))
      deriv(fx): fx * (1 - fx)
   tanh* = act:
      fn(x): tanh(x)
      deriv(fx): (let t = tanh(fx); 1 - t * t)
   relu* = act:
      fn(x): max(0, x)
      deriv(fx): (if fx == 0: 0 else: 1)
   leakyrelu* = act:
      fn(x): (if x < 0: 0.01 * x else: x)
      deriv(fx): (if fx < 0: 0.01 else: 1)
