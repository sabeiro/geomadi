---
title: "Routing"
author: Giovanni Marelli
date: 2019-10-22
rights:  Creative Commons Non-Commercial Share Alike 3.0
language: en-US
output: 
	md_document:
		variant: markdown_strict+backtick_code_blocks+autolink_bare_uris+markdown_github
---

# building a graph

We can download the street network from openstreetmap and the information are pretty detailed

![graph_detail](f_ops/graph_detail.png "graph detail")

_detail of a graph_

We see a lot of different street types, depending on the mean of transportation we need to run some operation on the graph and reduce the number of edges keeping the correct distances.

* download the graph (berlin: 162k edges, 147k nodes) (643k segments, 526k nodes)
* select the routable street classes (32k, 30k)
* simplify the graph (24k, 23k)
* take the largest connected subgraph (17k, 11k)
* project the graph
* weight the graph

depending on the mean of transportation we select only particular street classes

![graph by transportation](f_ops/graph_fastSlow.png "graph fast slow")

_different kind of graphs depending on the mean of transportation_

We build a graph from the geo dataframe

![graph by type](f_ops/graph_type.png "graph type")

_detail of a graph_

We label the nodes with geohash and depending on the digit used we have different number of nodes and connectivity

|digit|node|link|
|-|-|-|
|9|10k|22k|
|10|24k|29k|
|13|35k|32k|

With low digit we complete distort the geometry, with high number of digits we lose connectivity

![graph_digit](f_ops/graph_digit.png "graph detail")

_disconnected graph_

We realize that some parts are disconnected and therefore we take the largest connected graph

![graph_detail](f_ops/graph_disconnect.png "graph detail")

_disconnected graph_

We weight taking in consideration speed, street class, and length. We apply a factor for each street type

|highway|factor|
|-|-|
|motorway|3|
|primary|2|
|secondary|2|
|tertiary|1.5|
|residential|0.5|

We can than weight a graph with this simple formula:

$$ \frac{speed * type}{length} $$

and obtain a weighted graph

![graph_weight](f_ops/graph_weight.png "weighting graph")

_different weighting per street_

## calculating distance matrix

We selected the closest node per each spot

![graph_nearest](f_ops/graph_nearest.png "graph nearest")

_closest node per spot (in red)_

The first iterations show not logical routes which is mainly due to the direct nature of the graph


![graph_directed](f_ops/graph_directed.png "graph directed")

_shortest path between two spots in a directed graph_

A good directed graph is a lot of work and we by now use a undirected graph for reasonable routes

![graph_nearest](f_ops/graph_undirected.png "graph nearest")

_shortest path between two spots in a directed graph_

![graph_markov](f_ops/graph_markov.png "graph markov")

_changes in the Markov graph moving to weights_

We compare different graphs

![aymmetry_matrix](f_ops/asymmetry_matrix.png "")

_asymmetry matrix_

![aymmetry_distribution](f_ops/asymmetry_distribution.png "")

_asymmetry distribution_
