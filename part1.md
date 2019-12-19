# Introduction

Problem Statement:

-   computing accurate statistics like quantiles, or lower-counts on
    latency data

-   fully mergable across metrics and time

Traditional Approaches:

-   Compute statistics quantiles on each host, every minute and upstream the results as numeric metrics

-   Problem: What to do with the individual quantile values? Averaging is theoretically unsound, and results in larger
    errors.

Circonus has addressed this problem with the circllhist datastructure, that we describe in this document. It has been in
production use since 2010, used with many clients. Have talked about this at various conferences ..., blogs and
published an open source implementation. No academic paper has been published until now.

A very similar approach has been promoted and implemented by Gil Tene from Aszul Systems who, developed a High
Definition Range data-stsructure to capture latency data in Benchmarking applications (garbage collection latencies).

In the mean time, the awareness of this problem has gradually grown and other vendors in the Monitoring domain have
caught up:

-   Wavefront developed the t-digest data structure, and use it in their product.

-   Prometheus has added a basic histogram data structure, that aims at the same problem,

-   Most recently Data Dog has published a paper DD-Sketch that introduces a histogram data structure that is a very
    similar to our approach.

In this document we explain the circllhist data structure in detail and compare it to the above mentioned alternatives.

## Use Cases

Capture latency data from a large variety of sources including:

- Application metrics (request latencies, function call latencies, gc latencies)
- Kernel metrics (I/O latencies, scheduling latencies)
- Sensor data from embedded devices (audio data, pressure readings)

Aggregation across: 

- thousands of individual histogram metrics collected from different sources 
- time ranges from 10ms to years years

Powerful analytical capabilities:

- Information about the full distribution should be retained, so that general probabilistic modeling techniques can be
  applied.
  
- Efficient quantile computations, with general a-priori bounds on the relative error and low expected errors on real-world datasets.

- Precise counting of requests larger or lower than a given threshold.

# Related Work

In this section compare the following quantile sketches:

\begin{itemize}
\item[exact]
  Exact quantile estimation based on numpy arrays \url{numpy.org}
\item[prom]
  Quantile estimation based on \href{prometheus.io}{Prometheus} Histograms.
  We use a hand-written Python port of the original quantile functions written in go.
\item[hdr]
  The High Dynamic Range Histogram data-structure was introduced in \cite{hdr}.
  We use the Python implementation published at \url{https://github.com/HdrHistogram/HdrHistogram_py}.
\item[t-digest]
  The t-digest data structure was introduced in \cite{tdigest}.
  We use Java implementation by the original authors available at \url{github.com/tdunning/t-digest}.
\item[dd]
  The ``Distributed Distribution Sketch'' is a histogram datastructure that was introduced in \cite{dd}.
  We use the Python implementation published at \url{github.com/DataDog/sketches-py}
\item[circllhist]
  The Circonus Log-Linear Histogram datastructure described in this document.
  We use the Python/C implementation published at \url{github.com/circonus-labs/libcircllhist}.
\end{itemize}

The Prometheus monitoring system makes use of a very simple quantile sketch that consists of a list
of ``less than'' metrics, which count how many samples were inserted that are below manually
configured threshold values.
Prometheus Histograms are in wide use in practice.
The evaluation results are highly dependent on the number and location of the chosen thresholds.
We use the recommended number of around 10 threshold values, at locations which cover the whole
data range with emphasis on the likely quantile locations.
In practice these thresholds have to be chosen without knowledge of the dataset that is being recorded.

The HDR Histogram method was configured to resemble the circllhist method, with a range of
$10^{-128} .. 10^{128}$ and a accuracy of three significant digits.
