import random
from pyspark import SparkContext
from functools import partial

def newrand(x,sd):
	result=[]	
	for v in sd.value:
		random.seed(v)
		result.append(random.randint(1,10))
	return result


sc = SparkContext(appName="Simple App")
seed_numbers=range(10)
sd=sc.broadcast(seed_numbers)


rdd=sc.parallelize(range(1000)).map(partial(newrand, sd=sd))

print rdd.top(1)