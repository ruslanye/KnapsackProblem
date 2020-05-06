#!/usr/bin/env python3
import sys, random

f = "test.in"

if len(sys.argv)>1:
    f = sys.argv[1]
f = open(f, "w")
W = 10000
n = 256

f.write(str(n)+' '+str(W)+'\n')
for i in range(n):
    w = random.randint(1, 128)
    v = random.randint(1, 512)
    f.write(str(w)+' '+str(v)+'\n')
f.close()