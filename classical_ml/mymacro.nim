import std/macros

macro defineSchema(name, schema: untyped): untyped =
  result = newStmtList()
  echo schema.treerepr

defineSchema IrisClassifierSchema:
  properties:
    node(`ref` = node)
  definitions:
    node:
      oneOf:
        properties:
          class(`enum` = ["setosa", "versicolor", "virginica"], nullable = true)
        properties:
          feature(`enum` = ["sepal_length", "sepal_width", "petal_length", "petal_width"])
          threshold(`type` = number)
          left(`ref` = node)
          right(`ref` = node)

defineSchema UserSchema:
  properties:
    name(`type` = string)
    email(`type` = string, format = "email")
    age(`type` = number, optional = true)

defineSchema ProductSchema:
  properties:
    name(`type` = string)
    price(`type` = number)
    tags:
      items(`ref` = tag)
  definitions:
    tag:
      properties:
        name(`type` = string)
        color(`type` = string, `enum` = ["red", "green", "blue"])

defineSchema ComplexObjectSchema:
  properties(allowExtra = true):
    info:
      properties:
        name(`type` = string)
        address:
          properties:
            street(`type` = string)
            city(`type` = string)
            country(`type` = string)

defineSchema RectangleSchema:
  properties:
    topLeft(`ref` = point)
    bottomRight(`ref` = point)
  definitions:
    point:
      prefixItems(`type` = int, `type` = int)

defineSchema CatalogEntry:
  properties:
    id(`type` = int)
    description(`type` = string, nullable = true)
    metadata(`ref` = metaData, optional = true)
  definitions:
    metaData:
      properties:
        created(`type` = string, format = "date-time")
        createdBy(`type` = string)

defineSchema InfiniteLoop:
  definitions:
    loop(`ref` = loop)
  `ref` = loop
