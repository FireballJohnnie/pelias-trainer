# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# stream: Expert streamer (pinned store to device slots) and boundary store.
# pelias-deps: hal
pelias_module(stream
    CU expert_streamer.cu boundary_store.cu
    DEPS hal)
