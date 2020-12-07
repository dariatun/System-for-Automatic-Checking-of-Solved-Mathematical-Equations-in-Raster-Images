import sys

sum = 0
count = 0
for line in sys.stdin:
    if 'q' == line.rstrip():
        break
    sum += float(line)
    count += 1
print("Mean: " + str(sum / count))

print("Exit")