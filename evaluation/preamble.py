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

def pysize(o):
    "Return size of generic python serialization"
    return len(pickle.dumps(o))

#
# Merger Classes
#
class Merger(object):
    def insert(self, batch):
        raise NotImplementedError()

    def quantile(self, q):
        raise NotImplementedError()

    def bsize(self):
        raise NotImplementedError()

        
class ExactMerger(Merger):
    name = "exact"
    def __init__(self):
        self.data = np.array([])
    def insert(self, batch):
        self.data = np.concatenate([self.data, batch])
    def merge(self, other):
        self.data = np.concatenate([self.data, other.data])
    def quantile(self, q):
        self.data = np.sort(self.data)
        n = len(self.data)
        k = math.floor((n - 1) * q)
        return self.data[ k ]
        # return np.percentile(self.data, q*100, interpolation="linear")
    def bsize(self):
        return pysize(self.data)

class PromMerger(Merger):
    name = "prom"
    def __init__(self):
        self.X = np.array([ 0, 10, 100, 1000, 10000 ]) # thresholds
        self.N = len(self.X)
        self.C = np.array([0] * self.N) # counts
    def bsize(self):
        return pysize(self.C)
    def insert(self, batch):
        for x in batch:
            for i in range(N):
                if x < self.X[i]:
                    self.C[i] += 1
    def merge(self, other):
        self.X = self.X + other.X
    def quantile(self, q):
        rank = q * self.C[self.N - 1]
        b = np.searchsorted(self.C, rank, "right")
        if b == 0:
            return self.X[0]
        bucketStart = self.X[b - 1]
        bucketEnd   = self.X[b]
        bucketCount = self.C[b] - self.C[b - 1]
        bucketRank  = rank - self.C[b - 1]
        return bucketStart + (bucketEnd - bucketStart) * (bucketRank / bucketCount)
    
class CircllhistMerger(Merger):
    name = "circllhist"
    def __init__(self):
        self.H = Circllhist()
    def insert(self, batch):
        for v in batch: self.H.insert(v)
    def merge(self, other):
        self.H.merge(other.H)
    def quantile(self, q):
        return self.H.quantile(q, qtype=7)
    def bsize(self):
        # base64 encoding has 4/3 overhead compared to binary
        return len(self.H.to_b64())*3/4

class HdrMerger(Merger):
    name = "HdrHistogram"
    def __init__(self):
        self.H = HdrHistogram(1, 10**256, 3)
        self.M = 10**10 + 0.0
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

class TdigestMerger(Merger):
    name = "tdigest"
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

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import jnius_config
jnius_config.set_classpath('/tmp/t-digest-3.2.jar')
import jnius
from jnius import autoclass

class TdigestJavaMerger(Merger):
    name = "tdigest"
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

class DDMerger(Merger):
    name = "dd-sketch"
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
    for batch in batches:
        print(".",end="")
        o = cls()
        o.insert(batch)
        objects.append(o)
    t1 = time()
    print("")

    print("Merge ...")
    o_total_container = [ None ]
    def _merge():
        o_total = cls()
        for o in objects:
            o_total.merge(o)
        o_total_container[0] = o_total
    t_merge = perf(_merge)
    o_total = o_total_container[0]

    Q = {}
    print("Quantiles ...")
    def _quantiles():
        for q in quantiles:
            Q[q] = o_total.quantile(q)
    t_quantiles = perf(_quantiles)

    return {
        "bsize" : o_total.bsize(),
        "quantiles" : Q,
        "timings" : {
            "insert" : t1 - t0,
            "merge" : t_merge,
            "quantiles" : t_quantiles
        }
    }

METHODS = [ "exact",     "tdigest",     "hdr",     "dd",     "circllhist" ]
CLASSES = [ ExactMerger, TdigestJavaMerger, HdrMerger, DDMerger, CircllhistMerger ]

def evaluate_all(batches, quantiles):
    out = {}
    for i in range(len(METHODS)):
        name = METHODS[i]
        cls = CLASSES[i]
        print("# " + name)
        out[name] = evaluate(cls, batches, quantiles)
    return out

#
# Statistics
#
def fmtq(q):
    if q == 0: return "q0"
    if q == 1: return "q1"
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
            # stats_df[fmtq(q)] = y
            z = stats["exact"]["quantiles"][q]
            y = stats[name]["quantiles"][q]
            q_err.append( abs(y - z) / z * 100 )
        stats_dict[name] = q_err
    return pd.DataFrame(stats_dict, index=[fmtq(q) + "-err%" for q in quantiles ])

#
# Plotting Helper
#
def log_plot(X, Qs, xmin = 1e-1, xmax = 1e6):
    plt.figure(figsize=(20,5))
    ax = plt.subplot(1,1,1)
    ax.hist(X, bins=np.exp(np.linspace(np.log(xmin), np.log(xmax), 300)));
    ax.text(0.99, 0.95, '{} Requests'.format(len(X)), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)
    for y in Qs: ax.axvline(y, color="blue")
    plt.xscale("log")
    plt.xlim(xmin ,xmax)
    return ax

def lin_plot(X, Qs, xmin=0, xmax=110):
    plt.figure(figsize=(20,5))
    ax = plt.subplot(1,1,1)
    ax.hist(X, bins=np.linspace(xmin, xmax, 300));
    ax.text(0.99, 0.95, '{} Samples'.format(len(X)), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)
    for y in Qs: ax.axvline(y, color="blue")
    ax.set_ylabel("count")
    plt.xlim(xmin ,xmax)
    return ax

#
# Others
#
def save(s, path):
    with open(path, "w") as fh: fh.write(s)
