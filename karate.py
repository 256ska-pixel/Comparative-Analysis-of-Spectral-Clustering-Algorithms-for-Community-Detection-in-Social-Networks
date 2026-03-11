from scipy.io import mmread
from un import Un
import sklearn.cluster
import numpy

r = mmread(r"soc-karate.mtx")
a = r.toarray()
b = Un(a, 2)
print(b.ncsym())
print(b.ncrw())
print(b.unc())
print(b.modularitym(b.ncsym()))















