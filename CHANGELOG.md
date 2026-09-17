# Changelog

All notable changes to ALICE-Analytics will be documented in this file.

## [Unreleased]

## [0.1.1] - 2026-09-17

### Added
- `tests/analytic_oracle.rs` — 閉形式 / 2-pass 参照との突合 oracle 8 本 (CLAUDE.md § 解析解突合テスト規律、2026-09-17): Pébay 1-pass moment (mean / 分散 / g₁ / 超過尖度、順序独立)、R-7 quantile / percentile rank / Tukey fence、共分散行列 (対称・ρ = 1 / 0)、sliding window / SMA / EMA 閉形式 (1 − (1−α)ᵏ、α = 2/(span+1)) / change rate、streaming 回帰 (exact 直線 R² = 1、正規方程式)、HyperLogLog 3σ / **DDSketch 全 quantile ≤ α** (正負) / Count-Min ≥ 真値 + ε = e/w / heavy hitter / FNV-1a → fmix64、running median / MAD / EWMA・EWMV 漸化式、xorshift 決定性 / Laplace 2b² / randomized response の不偏推定

### Fixed (oracle 先行 red 1 → 修正 3 点)
- **`DDSketch::quantile` が bin の下端 γⁱ⁻¹ を返していた** → 相対誤差が最大 γ − 1 = 2α/(1−α) (α = 0.02 で 4 %、実測 3.1 %) と公開保証 α の 2 倍 → 代表値を `2γⁱ/(γ+1)` (両端から距離 α、Masson–Rim–Lee 2019) に、観測 [min, max] に clamp
- **負値の quantile が逆順**: negative bin を絶対値の小さい方から走査していたので q = 0 が最も 0 に近い負値を返していた → 絶対値の大きい方から
- **範囲外の値 (γ^(−offset) 未満 / γ^(bins−offset) 超) が count だけ増えて bin に入らず**、以後の quantile が 1 rank ずつずれていた → edge bin に収容 (保証は範囲内のみ)

### Fixed
- **`no_std` の `XorShift64::from_entropy()` が固定 seed を返す stub だった** — 差分プライバシー (Laplace / RandomizedResponse / RAPPOR) の noise が `no_std` では決定論になり privacy が silent に無効化されていた 削除し、entropy 由来の `new()` / `Default` / `default_params()` を `std` gate、`no_std` は caller が seed を渡す (`with_seed` / `with_probability` / 新設 `Rappor::with_seed_params`)
- clippy pedantic 137 件を 0 化し CI を `-W pedantic -D warnings` gate に (`math.rs` に整数 ↔ float の単一 audit 点 8 helper を集約、`inline(always)` 3 撤去、`From` / `try_from` 化)
- **`no_std` build が一度も通っていなかった** (`--no-default-features` で 38 error: f64 `mul_add` / `sqrt` / `ln` / `exp` / `powf` / `ceil` / `floor` / `round` + `std::f64::consts::LN_2`、6 module) — `src/math.rs` の `FloatExt` trait (`libm` 委譲、`std` 時は不使用) と `core::f64::consts` で修正、`export` の `MetricSnapshot` import を `std` gate host / bare-metal `thumbv7em-none-eabihf` / `x86_64` cross で build + clippy を確認

### Changed
- `libm = "0.2"` を依存に追加 (`no_std` build のみ使用、std build の依存 0 → 1 だが std 時は未使用) `no_std` の `sqrt` / `exp` 等は platform libm と最終 ulp で異なりうる (`src/math.rs` doc)
- CI: `no_std` job (host + thumbv7em + clippy) / `feature-powerset` (cargo-hack depth 2) / doc `-D warnings` を追加、`rust-toolchain.toml` (1.98.1 + thumbv7em target) 新設

## [0.1.0] - 2026-02-23

### Added
- `HyperLogLog` — cardinality estimation (10/12/14/16-bit precision variants)
- `DDSketch` — relative-error quantile estimation (128/256/512/1024/2048-bin variants)
- `CountMinSketch` — frequency estimation with configurable width and depth
- `HeavyHitters` — approximate top-K tracking (5/10/20 variants)
- `LaplaceNoise` / `RandomizedResponse` / `Rappor` — local differential privacy
- `PrivacyBudget` / `PrivateAggregator` — privacy budget tracking
- `MadDetector` / `ZScoreDetector` / `EwmaDetector` / `CompositeDetector` — streaming anomaly detection
- `MetricPipeline` — event-driven metric aggregation with ring buffer
- `MetricRegistry` — named metric registration and lookup
- `Mergeable` trait — all sketches support distributed merge
- `no_std` compatible core
- 44 tests (37 unit + 7 doc-test)
