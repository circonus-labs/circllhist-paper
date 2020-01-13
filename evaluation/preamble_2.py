import sys
import pickle
import json
from time import time
import timeit
import math

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd

sys.path.append("/opt/conda/lib/python3.6/site-packages/sketches-0.1-py3.6.egg")

from circllhist import Circllhist
from hdrh.histogram import HdrHistogram
from ddsketch.ddsketch import DDSketch
from tdigest import TDigest

# Configuration
plt.style.use(["ggplot"])
mpl.rcParams['text.color'] = "black"

#
# Others
#
def save(s, path):
    with open(path, "w") as fh: fh.write(s)

def save_results(stats, ds_name):
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError
    save(json.dumps(stats, default=convert), "results/" + ds_name + ".json")

def load(p):
    with open(p, "r") as fh:
        return json.load(fh)

def pysize(o):
    "Return size of generic python serialization"
    return len(pickle.dumps(o))

#
# Theoretical Quantile Functions
#
def quantile_type_1(X,q):
    n = len(X)
    r = q * n
    if q == 0:
        return X[0]
    else:
        return X[math.ceil(r) - 1]

def quantile_type_7(X, q):
    n = len(X)
    r = q*(n-1)
    return X[ math.floor(r) ]

def quantile_type_7i(X, q):
    n = len(X)
    r = q*(n-1)
    a = r - math.floor(r)
    return (1-a) * X[ math.floor(r) ] + a * X[ math.ceil(r) ]

def quantile_hdr(X, q):
    n = len(X)
    r = q*len(X)
    if r <= .5:
        return X[0]
    if r >= n - .5:
        return X[n - 1]
    t = r - .5
    return X[math.floor(t)]

def quantile_td(X, q):
    n = len(X)
    r = q*len(X)
    if r <= .5:
        return X[0]
    if r >= n - .5:
        return X[n - 1]
    t = r - .5
    a = t - math.floor(t)
    return (1-a) * X[math.floor(t)] + a * X[math.ceil(t)]

TQUANTILES = {
    "type-1" : quantile_type_1,
    "type-7" : quantile_type_7,
    "type-7i" : quantile_type_7i,
    "td" : quantile_td,
    "hdr" : quantile_hdr,
}

class ExactMerger(object):
    name = "exact"
    tquantile = quantile_type_7i
    def __init__(self):
        self.data = np.array([])
    def insert(self, batch):
        self.data = np.concatenate([self.data, batch])
    def merge(self, other):
        self.data = np.concatenate([self.data, other.data])
    def quantile(self, q):
        return np.percentile(self.data, q * 100)
    def bsize(self):
        return self.data.data.nbytes

class PromMerger(object):
    name = "prom1"
    thresholds = np.linspace(0, 200, 41)
    tquantile = quantile_type_1
    def __init__(self):
        # always include +inf bucket
        self.upper = np.append(self.thresholds, math.inf)
        self.N = len(self.upper)
        self.count = np.array([0] * self.N)
    def bsize(self):
        return self.count.data.nbytes
    def insert(self, batch):
        for x in batch:
            self.count += 1 * (self.upper >= x)
    def merge(self, other):
        self.count = self.count + other.count
    def quantile(self, q):
        N = self.N
        # q=0 Fix
        if q == 0:
            out = self.upper[0]
            i = 0
            while i < N and self.count[i] == 0:
                out = self.upper[i]
                i += 1
            return out
        rank = q * self.count[N - 1]
        b = np.searchsorted(self.count, rank, side="left")
        # C[b - 1] < rank <= C[b]
        if b == N - 1:
            return self.upper[N - 2]
        if b == 0 and self.upper[0] <= 0:
            return self.upper[0]
        bucketStart = 0
        bucketEnd   = self.upper[b]
        bucketCount = self.count[b]
        if b > 0:
            bucketStart =  self.upper[b - 1]
            bucketCount -= self.count[b - 1]
            rank        -= self.count[b - 1]
        return bucketStart + (bucketEnd - bucketStart) * (rank / bucketCount)

class CircllhistMerger(object):
    name = "circllhist"
    tquantile = quantile_type_1
    def __init__(self):
        self.H = Circllhist()
    def insert(self, batch):
        for v in batch: self.H.insert(v)
    def merge(self, other):
        self.H.merge(other.H)
    def quantile(self, q):
        return self.H.quantile(q)
    def bsize(self):
        # base64 encoding has 4/3 overhead compared to binary
        return len(self.H.to_b64())*3/4

class HdrMerger(object):
    name = "HdrHistogram"
    tquantile = quantile_hdr
    M = float(10**10)
    def __init__(self):
        self.H = HdrHistogram(1, 10**256, 2)    
    def insert(self, batch):
        for val in batch:
            self.H.record_value(val * self.M)
    def merge(self, other):
        self.H.add(other.H)
    def quantile(self, q):
        return self.H.get_value_at_percentile(q * 100) / self.M
    def bsize(self):
        # base64 encoding has 4/3 overhead compared to binary
        return len(self.H.encode()) * 3/4

class TdigestMerger(object):
    name = "tdigest"
    tquantile = quantile_td
    def __init__(self):
        self.td = TDigest(delta=0.05)
    def insert(self, batch):
        for val in batch:
            self.td.update(val, 1)
        self.td.compress()
    def merge(self, other):
        self.td = self.td + other.td
        self.td.compress()
    def quantile(self, q):
        return self.td.percentile(q * 100)
    def bsize(self):
        return pysize(self.td.to_dict())

import os, sys
if not "jnius" in sys.modules:
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    import jnius_config
    jnius_config.set_classpath('/tmp/t-digest-3.2.jar')
    import jnius
    from jnius import autoclass

class TdigestJavaMerger(object):
    name = "tdigest"
    tquantile = quantile_td
    def __init__(self):
        TD = autoclass('com.tdunning.math.stats.TDigest')
        # the compression parameter.  100 is a common value for normal uses. 
        self.td = TD.createDigest(100)
    def insert(self, batch):
        for val in batch:
            self.td.add(val, 1)
        self.td.compress()
    def merge(self, other):
        self.td.add(other.td)
        self.td.compress()
    def quantile(self, q):
        return self.td.quantile(q)
    def bsize(self):
        return self.td.byteSize()

class DDMerger(object):
    name = "dd-sketch"
    tquantile = quantile_type_7
    def __init__(self):
        self.dd = DDSketch(alpha=0.01)
    def insert(self, batch):
        for v in batch: self.dd.add(v)
    def merge(self, other):
        self.dd.merge(other.dd)
    def quantile(self, q):
        return self.dd.quantile(q)
    def bsize(self):
        return pysize(self.dd.store)

#
# Evaluation
#
def perf(f):
    timer = timeit.Timer("f()", globals = {"f": f})
    n, t = timer.autorange()
    durations = timer.repeat(5, n)
    return min(durations)/n

def evaluate(cls, batches, quantiles):
    t0 = time()
    objects = []
    print("Insert ", end="")
    count = 0.
    for batch in batches:
        print(".",end="")
        o = cls()
        o.insert(batch)
        objects.append(o)
        count += len(batch)
    t1 = time()
    t_insert = (t1 - t0) / count
    print("")

    print("Merge ...")
    o_total_container = [ None ]
    def _merge():
        o_total = cls()
        for o in objects:
            o_total.merge(o)
        o_total_container[0] = o_total
    t_merge = perf(_merge) / len(objects)
    o_total = o_total_container[0]

    Q = {}
    print("Quantiles ...")
    def _quantiles():
        for q in quantiles:
            Q[q] = o_total.quantile(q)
    t_quantiles = perf(_quantiles) / len(quantiles)

    # Compute exact quantiles
    # with theoretical quantile function on raw data
    X = np.concatenate( batches )
    X.sort()
    EQ = { q : cls.tquantile(X, q) for q in quantiles }

    return {
        "bsize" : o_total.bsize(),
        "quantiles" : Q,
        "exact_quantiles" : EQ,
        "timings" : {
            "insert" : t_insert,
            "merge" : t_merge,
            "quantiles" : t_quantiles
        }
    }

#           0           1            2                  3          4         5
METHODS = [ "exact",    "prom",      "tdigest",         "hdr",     "dd",     "circllhist" ]
CLASSES = [ ExactMerger, PromMerger, TdigestJavaMerger, HdrMerger, DDMerger, CircllhistMerger   ]

def evaluate_all(batches, quantiles):
    out = {}
    for i in range(len(METHODS)):
        name = METHODS[i]
        cls = CLASSES[i]
        print("# " + name)
        out[name] = evaluate(cls, batches, quantiles)
    return out

def reevaluate(STATS, name, batches, quantiles):
    for i in range(len(METHODS)):
        if METHODS[i] == name:
            cls = CLASSES[i]
            print("# " + name)
            STATS[name] = evaluate(cls, batches, quantiles)

#
# Statistics
#
def fmtq(q):
    if q == 0: return "q0"
    if q == 1: return "q1"
    if type(q) == str: return q
    return "q" + "{:.5g}".format(q).strip("0")

def stats_size(stats):
    return pd.DataFrame({ name : [ stats[name]["bsize"] ] for name in stats.keys() }, index = ["bsize"])

def stats_timing(stats):
    return pd.DataFrame({ name : stats[name]["timings"] for name in stats.keys() })

def stats_quantiles(stats):
    stats_dict = {}
    quantiles = sorted(stats["exact"]["quantiles"].keys())
    for name in stats.keys():
        stats_dict[name] = [ stats[name]["quantiles"][q] for q in quantiles ]
    return pd.DataFrame(stats_dict, index=[fmtq(q) for q in quantiles ])

def stats_qerr(stats):
    stats_dict = {}
    quantiles = sorted(stats["exact"]["quantiles"].keys())
    for name in stats.keys():
        q_err = []
        for q in quantiles:
            z = stats[name]["exact_quantiles"][q]
            y = stats[name]["quantiles"][q]
            q_err.append( abs(y - z) / z * 100 )
        stats_dict[name] = q_err
    return pd.DataFrame(stats_dict, index=[fmtq(q) for q in quantiles ])

def stats_quantile_df(stats):
    CIDX = []
    CDTA = []
    QS = sorted(list(stats['exact']['quantiles'].keys()))
    for name in stats:
        CDTA.append([ stats[name]['quantiles'][q] for q in QS ])
        CIDX.append([ name, "quantiles" ])
        CDTA.append([ stats[name]['exact_quantiles'][q] for q in QS ])
        CIDX.append([ name, "exact_quantiles" ])
    return pd.DataFrame(CDTA, index = pd.MultiIndex.from_tuples(CIDX), columns = [fmtq(q) for q in QS ]).transpose()
