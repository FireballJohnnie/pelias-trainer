# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# train: Optimizer, LR schedule, checkpoints, metrics, the trainer loop.
# pelias-deps: sched model data
pelias_module(train
    CC trainer.cc schedule.cc optimizer.cc checkpoint.cc metrics.cc
    DEPS sched model data)
