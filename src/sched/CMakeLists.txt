# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# sched: The only place that composes kernels into layers; the layer-major step, enqueue-only.
# pelias-deps: kern model stream
pelias_module(sched
    CU step.cu layer_fwd.cu layer_bwd.cu head.cu mtp.cu
    DEPS kern model stream)
