# Contributing to ALICE-Analytics

## Build

```bash
cargo build
cargo build --no-default-features   # no_std check
```

## Test

```bash
cargo test
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **no_std core**: all data structures must compile without `std`. Use fixed-size arrays.
- **Mergeable**: every sketch must implement the `Mergeable` trait for distributed aggregation.
- **Mathematical guarantees**: error bounds must be provable (e.g., HLL standard error = 1.04 / sqrt(m)).
- **Fixed memory**: sketch sizes are compile-time constants via generics or macros.
- **FNV-1a hashing**: use `FnvHasher` for deterministic, portable hashing across nodes.
- **Reciprocal constants**: pre-compute `1.0 / N` to avoid division in hot paths.
