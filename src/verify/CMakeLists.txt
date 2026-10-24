# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# verify: Gate runner, canaries, KL tools, dump comparison.
# pelias-deps: hal io
pelias_module(verify
    CC gate_runner.cc kl.cc dump_compare.cc
    CU canary.cu
    DEPS hal io)
