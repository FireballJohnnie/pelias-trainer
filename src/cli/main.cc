// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// cli/main.cc: entry point. The framework prints its name, version and backend and stops: no trainer code exists yet.
// --show-defaults prints the defaults of the author's ruling of 2026-10-24 (Mode Q, no extra bytes, planner automatic).
#include <cstdio>
#include <cstring>

#include "config/trainer_config.h"
#include "pelias/version.h"

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("%s\n", PELIAS_PRODUCT_NAME);
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--show-defaults") == 0) {
        std::printf("%s\n", PELIAS_PRODUCT_NAME);
        std::printf("%s", pelias::config::summary(pelias::config::default_config()).c_str());
        return 0;
    }
    std::printf("%s\n", PELIAS_PRODUCT_NAME);
    std::printf("Framework only: nothing is implemented. See docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md.\n");
    return 0;
}
