# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# intake: Persona, parameter and client sheets; import of premade lists; intake checks (counts only).
# pelias-deps: data tok
pelias_module(intake
    CC sheets.cc import.cc checks.cc clients.cc
    DEPS data tok)
