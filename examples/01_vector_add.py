#!/usr/bin/env python3
import numpy as np
import mini_pallas


@mini_pallas.kernel
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([5.0, 6.0, 7.0, 8.0])
out = np.zeros(4)

add_vectors_kernel(x, y, out)

print("Result:", out)  # [6. 8. 10. 12.]
print("\n--- IR ---")
print(add_vectors_kernel.show_ir())
print("\n--- Lowered NumPy Code ---")
print(add_vectors_kernel.lower())
