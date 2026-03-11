import sklearn.cluster
import numpy

class Un:
    def __init__(self, s, k):
        self.s = s
        self.k = k


    def unc(self):
        n = len(self.s)
        d = [[0 for i in range(n)] for i in range(n)]
        for i in range(n):
            d[i][i] = sum(self.s[i])
        la = numpy.subtract(d, self.s)
        f, g = numpy.linalg.eigh(la)
        g = numpy.array(g)
        u = [g[:, 0]]
        q = f[0]
        j = 1
        while len(u)!=self.k:
            if f[j] > q:
                q = f[j]
                u.append(g[:, j])
                j = j+1
            else:
                j = j+1
        u = numpy.array(u)
        u = numpy.matrix.transpose(u)
        km = sklearn.cluster.KMeans(n_clusters=self.k, random_state=0).fit(u)
        kml = km.labels_
        return kml


    def ncsym(self):
        n = len(self.s)
        d = [[0 for i in range(n)] for i in range(n)]
        for i in range(n):
            d[i][i] = sum(self.s[i])
        la = numpy.subtract(d, self.s)
        e = d
        for i in range(n):
            e[i][i] = numpy.sqrt(1/e[i][i])
        la = numpy.matmul(e, la)
        la = numpy.matmul(la, e)
        f, g = numpy.linalg.eigh(la)
        g = numpy.array(g)
        u = [g[:, 0]]
        q = f[0]
        j = 1
        while len(u) != self.k:
            if f[j] > q:
                q = f[j]
                u.append(g[:, j])
                j = j + 1
            else:
                j = j + 1
        u = numpy.array(u)
        u = numpy.matrix.transpose(u)
        for i in range(n):
            v = numpy.linalg.norm(u[i])
            for j in range(self.k):
                u[i][j] = u[i][j]/v
        km = sklearn.cluster.KMeans(n_clusters=self.k, random_state=0).fit(u)
        kml = km.labels_
        return kml


    def ncrw(self):
        n = len(self.s)
        d = [[0 for i in range(n)] for i in range(n)]
        for i in range(n):
            d[i][i] = sum(self.s[i])
        la = numpy.subtract(d, self.s)
        e = d
        for i in range(n):
            e[i][i] = 1/d[i][i]
        la = numpy.matmul(e, la)
        f, g = numpy.linalg.eig(la)
        h = numpy.argsort(f)
        f = numpy.array(f)[h]
        g = numpy.array(g)[h]
        g = numpy.array(g)
        u = [g[:, 0]]
        q = f[0]
        j = 1
        while len(u) != self.k:
            if f[j] > q:
                q = f[j]
                u.append(g[:, j])
                j = j + 1
            else:
                j = j + 1
        u = numpy.array(u)
        u = numpy.matrix.transpose(u)
        km = sklearn.cluster.KMeans(n_clusters=self.k, random_state=0).fit(u)
        kml = km.labels_
        return kml


    def modularitym(self, a):
        e = [[0 for i in range(self.k)] for i in range(self.k)]
        for i in range(self.k):
            for j in range(self.k):
                for l in range(len(a)):
                    if a[l] == i:
                        for m in range(len(a)):
                            if a[m] == j:
                                e[i][j] = e[i][j]+self.s[l][m]
        for i in range(self.k):
            e[i][i] = e[i][i]/2
        s = [sum(self.s[i]) for i in range(len(self.s))]
        su = sum(s)/2
        for i in range(self.k):
            for j in range(self.k):
                e[i][j] = e[i][j]/su
        m = [e[i][i]-(sum(e[i])*sum(e[i])) for i in range(self.k)]
        mod = sum(m)
        return mod
































