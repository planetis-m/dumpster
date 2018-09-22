# Encapsulated family of Algorithms
type
   IBrakeBehavior = ref object of RootObj
   BrakeWithABS = ref object of IBrakeBehavior
   Brake = ref object of IBrakeBehavior

method brake(this: IBrakeBehavior) {.base.} = discard

method brake(this: BrakeWithABS) =
   echo("Brake with ABS applied")

method brake(this: Brake) =
   echo("Simple Brake applied")

type
   Car = ref object of RootObj
      # Client that can use the algorithms above interchangeably
      brakeBehavior: IBrakeBehavior

proc applyBrake(this: Car) =
   this.brakeBehavior.brake()

proc setBrakeBehavior(this: Car, brakeType: IBrakeBehavior) =
   this.brakeBehavior = brakeType

type
   Sedan = ref object of Car
      # Client 1 uses one algorithm (Brake) in the constructor

   Suv = ref object of Car
      # Client 2 uses another algorithm (BrakeWithABS) in the constructor

proc newSedan(): Sedan =
   new(result)
   result.brakeBehavior = Brake()

proc newSuv(): Suv =
   new(result)
   result.brakeBehavior = BrakeWithABS()

# Using the Car example
proc main() =
   let sedanCar = newSedan()
   sedanCar.applyBrake() # This will invoke class Brake

   let suvCar = newSuv()
   suvCar.applyBrake()   # This will invoke class BrakeWithABS

   # set brake behavior dynamically
   suvCar.setBrakeBehavior(Brake())
   suvCar.applyBrake()   # This will invoke class Brake

main()
