import os

print(os.listdir(os.getcwd()))

with open("ecc_v30_combined.csv", 'r') as f:
    lines = f.readlines()

accum = 0
for l in lines[1:]:
    bin, count = l.split(",")

    x = int(count)
    accum += x
    print(bin, accum)
