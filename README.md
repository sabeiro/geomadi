---
title: "Geomadi library description"
author: Giovanni Marelli
date: 2019-01-12
rights:  Creative Commons Non-Commercial Share Alike 3.0
language: en-US
output: 
	md_document:
		variant: markdown_strict+backtick_code_blocks+autolink_bare_uris+markdown_github
---

# geomadi

geomadi is a library for spatial operations

![module_geomadi](docs/f_ops/module_geomadi.svg "module geomadi")

_overview of geomadi modules_

## demand prediction

[demand prediction](docs/activation.md)

* theoretical activation potential
* aggregation of demand and offer on geohash

![demand_prediction](docs/f_ops/map_urev.png "unit revenue")

## location 

* enrichment of geo features on geohash
* prediction on single geohash

[spatial operations and geographical transformation](docs/location.md)

![pop_dens](docs/f_ops/popDens_interp.png "population density")

## evaluate operations metrics

* restructuring data based on quantities of interest
* motion patterns

[evaluates operations](docs/ride.md)

![ride_map](docs/f_ops/ride_map.png "ride maps")

## optimization engine

* Monte Carlo and Markov chains
* convergent solutions

[optimization engine](docs/optimization.md)

![opt route](docs/f_ops/opt_small_02.png "optimization route")

## dynamics of motion

[dynamics of motion](docs/motion.md)

* from coordinates to dynamic of motion
* city mobility

![motion](docs/f_ops/dwelling_city.png "dwelling city")

## routing and graphs

[routing and graphs](docs/routing.md)

* building an efficient graph based on mean of transportation
* finding the most optimal path and calculating weighting matrices

![motion](docs/f_ops/graph_weight.png "dwelling city")


