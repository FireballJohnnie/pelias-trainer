// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// Status codes returned across module boundaries (master 5.1: status codes and RAII wrappers; no exceptions across the
// kernel-launch layer).
#pragma once

namespace pelias {

enum class Status {
    Ok = 0,
    InvalidArgument,
    NotFound,
    IoError,
    CudaError,
    OutOfMemory,
    NotImplemented,
    Aborted,
};

}  // namespace pelias
