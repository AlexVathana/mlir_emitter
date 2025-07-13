def emit_mlir(tensors, operation):
    size = tensors[0].shape[0]
    dtype = tensors[0].dtype

    lines = []

    lines.append("module {")
    lines.append(f"  func.func @tensor_add(%A: memref<{size}x{dtype}>, %B: memref<{size}x{dtype}>, %C: memref<{size}x{dtype}>) {{")
    lines.append(f"    %c0 = arith.constant 0 : index")
    lines.append(f"    %c1 = arith.constant 1 : index")
    lines.append(f"    %c{size} = arith.constant {size} : index")
    lines.append(f"    scf.for %i = %c0 to %c{size} step %c1 {{")
    lines.append(f"      %a = memref.load %A[%i] : memref<{size}x{dtype}>")
    lines.append(f"      %b = memref.load %B[%i] : memref<{size}x{dtype}>")
    lines.append(f"      %r = arith.addf %a, %b : {dtype}")
    lines.append(f"      memref.store %r, %C[%i] : memref<{size}x{dtype}>")
    lines.append(f"    }}")
    lines.append(f"    return")
    lines.append(f"  }}")
    lines.append("}")
    return "\n".join(lines)