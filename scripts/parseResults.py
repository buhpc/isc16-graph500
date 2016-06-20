import re
from sys import argv

header = re.compile("[a-zA-Z\ ]+")
pattern = re.compile("\d+.\d+")
sums = {'Construct Graph':0, 'BFS':0}
count = {'Construct Graph':0, 'BFS':0}

for idx in xrange(1, len(argv)):
  with open(argv[idx]) as file:
    for line in file:
      if re.search(header, line):
        event = re.search(header, line).group(0)
        time = pattern.findall(line)
        sums[event] += float(time[0]) 
        count[event] += 1

# print results
print "Results for graph construction:"
avg = sums['Construct Graph'] / count['Construct Graph']
print "avg time = %f secs" % (avg)

print "\nResults for breadth first search"
avg = sums['BFS'] / count['BFS']
print "avg time = %f secs" % (avg)
