# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# plan: Reads the probe constants, solves the plan, prints it before any allocation.
# pelias-deps: hal
pelias_module(plan
    CC probe_constants.cc roofline_solver.cc memory_plan.cc
    DEPS hal)
