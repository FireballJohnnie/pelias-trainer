# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# hal: Device selection, stream/event wrappers, arena allocator, pinned pool, VRAM governor, sync audit.
# pelias-deps: 
pelias_module(hal
    CC device.cc stream.cc event.cc pinned_pool.cc vram_governor.cc sync_audit.cc
    CU arena.cu)
