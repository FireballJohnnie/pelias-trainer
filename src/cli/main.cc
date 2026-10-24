// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// cli/main.cc: entry point. The framework prints its name, version and backend and stops: no trainer code exists yet.
#include <cstdio>
#include <cstring>

#include "pelias/version.h"

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("%s\n", PELIAS_PRODUCT_NAME);
        return 0;
    }
    std::printf("%s\n", PELIAS_PRODUCT_NAME);
    std::printf("Framework only: nothing is implemented. See docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md.\n");
    return 0;
}
