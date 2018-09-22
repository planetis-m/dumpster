type
   BillingStrategy = ref object of RootObj

   NormalStrategy = ref object of BillingStrategy
      # Normal billing strategy (unchanged price)

   HappyHourStrategy = ref object of BillingStrategy
      # Strategy for Happy hour (50% discount)

   Customer = ref object
      drinks: seq[float]
      strategy: BillingStrategy

using
   strategy: BillingStrategy
   rawPrice, price: float
   quantity: int

method getActPrice(this: BillingStrategy, rawPrice): float {.base.} =
   discard

method getActPrice(this: NormalStrategy, rawPrice): float =
   rawPrice

method getActPrice(this: HappyHourStrategy, rawPrice): float =
   rawPrice * 0.5

proc add(this: Customer, price, quantity) =
   this.drinks.add(this.strategy.getActPrice(price * quantity.float))

proc newCustomer(strategy): Customer =
   new(result)
   result.strategy = strategy

proc setStrategy(this: Customer, strategy) =
   this.strategy = strategy

proc printBill(this: Customer) =
   # Payment of bill
   var sum = 0.0
   for i in this.drinks:
      sum += i
   echo("Total due: ", sum)
   this.drinks.setLen(0)

proc main() =
   # Prepare strategies
   let normalStrategy = NormalStrategy()
   let happyHourStrategy = HappyHourStrategy()

   let firstCustomer = newCustomer(normalStrategy)

   # Normal billing
   firstCustomer.add(1.0, 1)

   # Start Happy Hour
   firstCustomer.setStrategy(happyHourStrategy)
   firstCustomer.add(1.0, 2)

   # New Customer
   let secondCustomer = newCustomer(happyHourStrategy)
   secondCustomer.add(0.8, 1)
   # The Customer pays
   firstCustomer.printBill()

   # End Happy Hour
   secondCustomer.setStrategy(normalStrategy)
   secondCustomer.add(1.3, 2)
   secondCustomer.add(2.5, 1)
   secondCustomer.printBill()

main()
