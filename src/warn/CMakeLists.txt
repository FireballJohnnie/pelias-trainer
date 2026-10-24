# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# warn: Cautions about fit, loops and a model that does not work; the client matrix.
# pelias-deps: data verify intake
pelias_module(warn
    CC fit.cc loops.cc works.cc client_matrix.cc
    DEPS data verify intake)
