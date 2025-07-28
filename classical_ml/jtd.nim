import jsonpak, jsonpak/[mapper, dollar]

const
  jtypdef = %*{
    "definitions": {
      "node": {
        "optionalProperties": {
          "class": {"enum": ["setosa", "versicolor", "virginica"]},
          "feature": {"enum": ["sepal_length", "sepal_width", "petal_length", "petal_width"]},
          "threshold": {"type": "float32"},
          "left": {"ref": "node"},
          "right": {"ref": "node"}
        }
      }
    },
    "properties": {
      "node": {"ref": "node"}
    }
  }

echo jtypdef

# var bitset = 0'u32
# if nullable:
#   bitset = bitset or Nullable.uint32
# if optional:
#   bitset = bitset or Optional.uint32
# if additional:
#   bitset = bitset or Additional.uint32
