# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# startup: Start-up quick pass, machine profile, scan of what holds GPU memory.
# pelias-deps: hal plan io project
pelias_module(startup
    CC quick_pass.cc machine_profile.cc vram_scan.cc
    DEPS hal plan io project)
