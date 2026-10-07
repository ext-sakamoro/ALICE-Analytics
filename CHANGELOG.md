# Changelog

All notable changes to ALICE-Analytics will be documented in this file.

## [Unreleased]

### Added
- `law` feature と `law` module: `alice_zip::law::SignalLaw` に対する点集合の残差 `y − f(x)` を `DDSketch2048` に流し込み、`p50` / `p90` / `p99` (sketch の相対誤差 α 以内)、`mean` / `min` / `max` / `max_abs` (厳密値)、件数を返す `residual_summary` と、ストリーミング版 `ResidualSketch` (`push` / `extend` / `quantile` / `summary`、α とスケールを指定可能)
  - law の成立範囲外 (または非有限) の `x` は評価せず `out_of_range` に数える (外挿しない)、`y − f(x)` が NaN / 無限大の点は `non_finite` に数えて要約に入れない
  - 残差は 2 の冪のスケール (既定は law 自身の残差 RMS 以下で最大の 2 の冪) で割って挿入し、保証範囲 `accurate_range` と範囲外の件数 `outside_accuracy` を報告する
  - `alice-zip = "0.5.1"` (`default-features = false`) を optional 依存に追加
- `DDSketch*::accurate_range()`: 相対誤差の保証が成り立つ大きさの範囲 `[γ^(−offset), γ^(BINS − 1 − offset)]`
- `examples/residual_summary.rs`
- `tests/law_residual.rs` (閉形式の順序統計量との突合 18 本) と `tests/analytic_oracle.rs` に `accurate_range` の範囲と境界外での上限不成立の検査
- `README_JP.md`、`scripts/docs_lint.py` (公開文書の語彙、CHANGELOG 構造、README の例と crate doctest の一致) と `scripts/test_docs_lint.py`、`scripts/stub_guard.sh`
- crate レベルの doctest (README の最初の例と同一)

### Changed
- `rust-version = "1.87"` を宣言 (1.86 では `is_multiple_of` が未安定でビルドできないことを実測)
- CI: test を 4 OS (macOS / Linux x86_64 / Linux arm64 / Windows) に、`law` 有りの `no_std` ビルド、MSRV、rustdoc (既定 / 全 feature)、docs lint (3 OS)、example 実行を追加 security-audit は cargo audit の DB を `target/` 配下に置き、install を `taiki-e/install-action` に変更、`scripts/preflight.sh` を CI と同じ引数に揃え `--quick` で `cargo test --lib` を実行
- 何もしない composite action を削除、deny.toml の license 許可を依存グラフに存在するもの (MIT / Apache-2.0) に限定し wildcard と未知の registry / git を deny
- README を全面改稿 (ライセンス表記を Cargo.toml の `MIT OR Apache-2.0` に一致させ、存在しない bridge module の記述を削除)
- `DDSketch` の範囲外の値に関するコメントの上端を `accurate_range()` に合わせて訂正

## [0.1.1] - 2026-09-17

### Added
- `tests/analytic_oracle.rs` — 閉形式 / 2-pass 参照との突合 oracle 8 本: Pébay 1-pass moment (mean / 分散 / g₁ / 超過尖度、順序独立)、R-7 quantile / percentile rank / Tukey fence、共分散行列 (対称・ρ = 1 / 0)、sliding window / SMA / EMA 閉形式 (1 − (1−α)ᵏ、α = 2/(span+1)) / change rate、streaming 回帰 (exact 直線 R² = 1、正規方程式)、HyperLogLog 3σ / **DDSketch 全 quantile ≤ α** (正負) / Count-Min ≥ 真値 + ε = e/w / heavy hitter / FNV-1a → fmix64、running median / MAD / EWMA・EWMV 漸化式、xorshift 決定性 / Laplace 2b² / randomized response の不偏推定

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
