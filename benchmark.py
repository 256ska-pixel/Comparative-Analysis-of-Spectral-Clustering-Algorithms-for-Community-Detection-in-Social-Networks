from networkx.generators.community import LFR_benchmark_graph
from sklearn.metrics.cluster import normalized_mutual_info_score as ni
import networkx as nx
from un import Un
import numpy


def lab(o, n):
    p = [0 for i in range(n)]
    for i in range(len(o)):
        for j in range(len(o[i])):
            p[o[i][j]] = i
    return p


def nmim(w):

    a1 = LFR_benchmark_graph(50,2.1,1.5,w,
                             average_degree=3,max_degree=9,
                             min_community=6,max_community=23,seed=10)
    c1 = {frozenset(a1.nodes[v]['community']) for v in a1}
    b1 = nx.to_numpy_array(a1,nodelist=[i for i in range(50)])
    d1 = [list(x) for x in c1]
    e1 = lab(d1,50)

    a2 = LFR_benchmark_graph(100,2.1,1.5,w,
                             average_degree=4,max_degree=12,
                             min_community=3,max_community=24,seed=10)
    c2 = {frozenset(a2.nodes[v]['community']) for v in a2}
    b2 = nx.to_numpy_array(a2,nodelist=[i for i in range(100)])
    d2 = [list(x) for x in c2]
    e2 = lab(d2,100)

    a3 = LFR_benchmark_graph(150,2.1,1.5,w,
                             average_degree=6,max_degree=19,
                             min_community=4,max_community=34,seed=10)
    c3 = {frozenset(a3.nodes[v]['community']) for v in a3}
    b3 = nx.to_numpy_array(a3,nodelist=[i for i in range(150)])
    d3 = [list(x) for x in c3]
    e3 = lab(d3,150)

    a4 = LFR_benchmark_graph(200,2.1,1.5,w,
                             average_degree=8,max_degree=26,
                             min_community=5,max_community=45,seed=10)
    c4 = {frozenset(a4.nodes[v]['community']) for v in a4}
    b4 = nx.to_numpy_array(a4,nodelist=[i for i in range(200)])
    d4 = [list(x) for x in c4]
    e4 = lab(d4,200)

    a5 = LFR_benchmark_graph(250,2.1,1.5,w,
                             average_degree=10,max_degree=33,
                             min_community=6,max_community=56,seed=10)
    c5 = {frozenset(a5.nodes[v]['community']) for v in a5}
    b5 = nx.to_numpy_array(a5,nodelist=[i for i in range(250)])
    d5 = [list(x) for x in c5]
    e5 = lab(d5,250)

    a6 = LFR_benchmark_graph(300,2.1,1.5,w,
                             average_degree=12,max_degree=40,
                             min_community=9,max_community=75,seed=10)
    c6 = {frozenset(a6.nodes[v]['community']) for v in a6}
    b6 = nx.to_numpy_array(a6,nodelist=[i for i in range(300)])
    d6 = [list(x) for x in c6]
    e6 = lab(d6,300)

    a7 = LFR_benchmark_graph(350,2.1,1.5,w,
                             average_degree=15,max_degree=48,
                             min_community=8,max_community=78,seed=10)
    c7 = {frozenset(a7.nodes[v]['community']) for v in a7}
    b7 = nx.to_numpy_array(a7,nodelist=[i for i in range(350)])
    d7 = [list(x) for x in c7]
    e7 = lab(d7,350)

    a8 = LFR_benchmark_graph(400,2.1,1.5,w,
                             average_degree=17,max_degree=55,
                             min_community=9,max_community=89,seed=10)
    c8 = {frozenset(a8.nodes[v]['community']) for v in a8}
    b8 = nx.to_numpy_array(a8,nodelist=[i for i in range(400)])
    d8 = [list(x) for x in c8]
    e8 = lab(d8,400)

    a9 = LFR_benchmark_graph(450,2.1,1.5,w,
                             average_degree=19,max_degree=62,
                             min_community=12,max_community=108,seed=10)
    c9 = {frozenset(a9.nodes[v]['community']) for v in a9}
    b9 = nx.to_numpy_array(a9,nodelist=[i for i in range(450)])
    d9 = [list(x) for x in c9]
    e9 = lab(d9,450)

    a10 = LFR_benchmark_graph(500,2.1,1.5,w,
                              average_degree=21,max_degree=69,
                              min_community=11,max_community=111,seed=10)
    c10 = {frozenset(a10.nodes[v]['community']) for v in a10}
    b10 = nx.to_numpy_array(a10,nodelist=[i for i in range(500)])
    d10 = [list(x) for x in c10]
    e10 = lab(d10,500)

    f1 = Un(numpy.array(b1),len(d1))
    f2 = Un(numpy.array(b2),len(d2))
    f3 = Un(numpy.array(b3),len(d3))
    f4 = Un(numpy.array(b4),len(d4))
    f5 = Un(numpy.array(b5),len(d5))
    f6 = Un(numpy.array(b6),len(d6))
    f7 = Un(numpy.array(b7),len(d7))
    f8 = Un(numpy.array(b8),len(d8))
    f9 = Un(numpy.array(b9),len(d9))
    f10 = Un(numpy.array(b10),len(d10))

    nmsym = numpy.array(
        [ni(f1.ncsym(),e1),ni(f2.ncsym(),e2),ni(f3.ncsym(),e3),ni(f4.ncsym(),e4),
         ni(f5.ncsym(),e5),ni(f6.ncsym(),e6),ni(f7.ncsym(),e7),ni(f8.ncsym(),e8),
         ni(f9.ncsym(),e9),ni(f10.ncsym(),e10)])

    nmrw = numpy.array(
        [ni(f1.ncrw(),e1),ni(f2.ncrw(),e2),ni(f3.ncrw(),e3),ni(f4.ncrw(),e4),
         ni(f5.ncrw(),e5),ni(f6.ncrw(),e6),ni(f7.ncrw(),e7),ni(f8.ncrw(),e8),
         ni(f9.ncrw(),e9),ni(f10.ncrw(),e10)])

    nmun = numpy.array(
        [ni(f1.unc(),e1),ni(f2.unc(),e2),ni(f3.unc(),e3),ni(f4.unc(),e4),
         ni(f5.unc(),e5),ni(f6.unc(),e6),ni(f7.unc(),e7),ni(f8.unc(),e8),
         ni(f9.unc(),e9),ni(f10.unc(),e10)])

    modsym = numpy.array(
        [f1.modularitym(f1.ncsym()),f2.modularitym(f2.ncsym()),
         f3.modularitym(f3.ncsym()),f4.modularitym(f4.ncsym()),
         f5.modularitym(f5.ncsym()),f6.modularitym(f6.ncsym()),
         f7.modularitym(f7.ncsym()),f8.modularitym(f8.ncsym()),
         f9.modularitym(f9.ncsym()),f10.modularitym(f10.ncsym())])

    modrw = numpy.array(
        [f1.modularitym(f1.ncrw()),f2.modularitym(f2.ncrw()),
         f3.modularitym(f3.ncrw()),f4.modularitym(f4.ncrw()),
         f5.modularitym(f5.ncrw()),f6.modularitym(f6.ncrw()),
         f7.modularitym(f7.ncrw()),f8.modularitym(f8.ncrw()),
         f9.modularitym(f9.ncrw()),f10.modularitym(f10.ncrw())])

    modun = numpy.array(
        [f1.modularitym(f1.unc()),f2.modularitym(f2.unc()),
         f3.modularitym(f3.unc()),f4.modularitym(f4.unc()),
         f5.modularitym(f5.unc()),f6.modularitym(f6.unc()),
         f7.modularitym(f7.unc()),f8.modularitym(f8.unc()),
         f9.modularitym(f9.unc()),f10.modularitym(f10.unc())])

    return nmsym,nmrw,nmun,modsym,modrw,modun


print(nmim(0.25))

nmsym, nmrw, nmun, modsym, modrw, modun = nmim(0.25)

print(nmsym, nmrw, nmun, modsym, modrw, modun)

import matplotlib.pyplot as plt

sizes = [50,100,150,200,250,300,350,400,450,500]

plt.plot(sizes, nmsym, label="NC Symmetric")
plt.plot(sizes, nmrw, label="Random Walk")
plt.plot(sizes, nmun, label="Unnormalized")

plt.xlabel("Network size")
plt.ylabel("NMI")
plt.legend()
plt.show()