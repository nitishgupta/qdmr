import re

pattern = re.compile("that (are|was|were) at least [0-9]+$")
r1 = pattern.match("that were at least 8")
r2 = pattern.match("that were at least 8.4 yards")

# r3 = pattern.match("that are atleast 20 yards and")
print(r1)
print(r2)

r1 = pattern.fullmatch("that were at least 8")
r2 = pattern.fullmatch("that were at least 8.4 yards")
print(r1)
print(r2)
# print(r3)


