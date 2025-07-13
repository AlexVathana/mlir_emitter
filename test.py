from dsl import Tensor, Add
from emitter import emit_mlir

A = Tensor("A", shape=(1024, ))
B = Tensor("B", shape=(1024, ))
C = Tensor("C", shape=(1024, ))

op = Add(A, B, C)

mlir_code = emit_mlir([A,B,C], op)
print(mlir_code)