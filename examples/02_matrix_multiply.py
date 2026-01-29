#!/usr/bin/env python3
import numpy as np
import mini_pallas

@mini_pallas.kernel
def matmul_kernel(x_ref, y_ref, o_ref):
  a, b = x_ref[...], y_ref[...]
  o_ref[...] = a @ b

if __name__ == "__main__":
  x = np.array([[1.0, 2.0], [3.0, 4.0]])
  y = np.array([[5.0, 6.0], [7.0, 8.0]])
  out = np.zeros((2, 2))
  
  matmul_kernel(x, y, out)
  print(out)
