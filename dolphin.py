from scipy.io import mmread
from un import Un
import sklearn.cluster

r = mmread(r"soc-dolphins.mtx")
a = r.toarray()
b = Un(a, 2)
print(b.ncsym())
print(b.ncrw())
print(b.unc())
c = sklearn.cluster.SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(a)
print(c.labels_)
