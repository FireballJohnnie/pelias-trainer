# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# flow: Propose, confirm, apply to a copy, re-test, final summary, plan card, Expert mode. Logic only; no interface.
# pelias-deps: plan startup project intake warn verify help
pelias_module(flow
    CC proposal.cc summary.cc expert.cc plan_card.cc
    DEPS plan startup project intake warn verify help)
