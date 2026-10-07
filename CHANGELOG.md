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
- **決定論 (プラットフォーム間の bit 一致)**: 超越関数 (`ln` / `exp` / `powf`) を `alice-det-math` 0.3 (`default-features = false`、`std` feature は本 crate の `std` に連動) 経由に変更 プラットフォームの `libm` は最後の ulp が macOS / glibc / MSVC / wasm で異なるため、sketch を複数マシンで merge する / テレメトリを再生する / 監査が生イベントから分位点を再計算する経路で推定値が割れていた `std` ビルドと `no_std` ビルドも bit まで一致する
  - `clippy.toml` (新規): `f32` / `f64` の inherent な超越関数 48 個 + `powi` 2 個を `disallowed-methods` に登録 CI の `cargo clippy --all-targets --all-features -- -D warnings` が red gate になる `sqrt` と `mul_add` は IEEE 754 が正確丸めを要求するので意図的に対象外 (理由を `clippy.toml` の冒頭に記載)
  - `src/math.rs`: 結合順を固定した整数冪 `ipow64` (繰り返し二乗) を追加 `DDSketch` の bin 代表値と `accurate_range()` が使う `γⁿ` に適用 `powi` は乗算木の結合順が未規定、`powf64` は `|y| = 1535` で相対誤差 8.9e-14 まで出るのに対し `ipow64` は 0 (`powi` と bit 一致、実測) `FloatExt` からは `ln` / `exp` / `powf` を削除し、`no_std` でも `libm` の超越関数を使えないようにした
- `tests/determinism_golden.rs` (7 本): `sketch` / `stats` / `window` / `anomaly` / `privacy` / `streaming_ops` / `law` の出力を bit 単位で直列化して SHA-256 を定数と突合 各シナリオは直列化 byte 数が 0 でないことを先に assert する (モジュールを実行しなくなったシナリオが空バッファのハッシュで通るのを防ぐ) 変異 12 通り (det-math を platform libm に戻す / `ipow64` を `powf` に戻す / 係数・符号・境界を変える / ガードを外す) が全て red になることを実測
- `tests/panic_contract.rs` (25 本): 退化入力の契約試験 空入力 / 定義域外 / 非有限 / `u64::MAX` 近傍の時刻 / 幅 0 の const generic に対して「値」「飽和値」「panic」のどれが正かを明示して assert `should_panic` は `expected` 付き 「panic しない」だけを assert する空振りを避けるため値で突合する
- `README.md` / `README_JP.md` に `## Determinism` / `## 決定論` 節 (層別の根拠、2 層の機械強制、保証の外、`claim-test` 注記)

### Changed
- `rust-version = "1.87"` を宣言 (1.86 では `is_multiple_of` が未安定でビルドできないことを実測)
- CI: test を 4 OS (macOS / Linux x86_64 / Linux arm64 / Windows) に、`law` 有りの `no_std` ビルド、MSRV、rustdoc (既定 / 全 feature)、docs lint (3 OS)、example 実行を追加 security-audit は cargo audit の DB を `target/` 配下に置き、install を `taiki-e/install-action` に変更、`scripts/preflight.sh` を CI と同じ引数に揃え `--quick` で `cargo test --lib` を実行
- 何もしない composite action を削除、deny.toml の license 許可を依存グラフに存在するもの (MIT / Apache-2.0) に限定し wildcard と未知の registry / git を deny
- README を全面改稿 (ライセンス表記を Cargo.toml の `MIT OR Apache-2.0` に一致させ、存在しない bridge module の記述を削除)
- `DDSketch` の範囲外の値に関するコメントの上端を `accurate_range()` に合わせて訂正
- `Cargo.toml`: `std` feature が `alice-det-math/std` を伝播する `[lints.clippy]` に `suboptimal_flops` / `imprecise_flops` の `allow` を理由付きで追加 (det-math 経由の呼出を `mul_add` / platform `libm` の形に書き戻す提案を crate の性質として 1 箇所で止める、`alice-det-math` 自身と同じ形) `[dev-dependencies]` に `sha2` (golden ハッシュ用)
- `tests/analytic_oracle.rs` / `tests/law_residual.rs` / `examples/residual_summary.rs`: 禁止した inherent メソッド (`powf` / `exp` / `powi`) の呼出を `alice_det_math` 経由と結合順固定の整数冪に置換 期待値は変わらない (`ipow64` は `powi` と bit 一致、実測)
- README の `no_std` の浮動小数点に関する記述を訂正 (超越関数は `std` の有無に関わらず `alice-det-math` 経由で、`libm` を使うのは `sqrt` / 丸め / `mul_add` だけ) 関連 crate に `alice-det-math` を追加

### Fixed
- **`DDSketch::insert` が非有限 / 極大の大きさで bucket index を overflow させていた** — `bucket_index` の `f64_i32(...) + self.offset` は、`value` が `inf` (上流の 0 除算が届いた場合) や `alpha = 0` (γ = 1 ⇒ `inv_ln_gamma` が無限大) のとき `as i32` が `i32::MAX` に飽和した上で offset を足すため i32 を溢れる debug ビルドでは panic し、**release ビルドでは負に wrap して直後の `max(0)` で bin 0 に入っていた** (= 2026-09-17 に修正した「範囲外の値が rank をずらす」と同型の silent な破損) `saturating_add` にして、doc の既存契約どおり上端 / 下端の edge bin に収容する
- **`TumblingWindow` / `HierarchicalRollup` が `u64::MAX` 近傍の時刻で overflow していた** — `current_start + window_ms` が debug では panic、release では wrap して `end_ms < start_ms` の結果を出していた `saturating_add` に変更 イベント数の計上は変わらない
- `RingBuffer::capacity()` が `N = 0` で underflow、`is_full()` が `% N` で 0 除算していた `N.saturating_sub(1)` と `N <= 1` の早期 return にして read 系アクセサを全域化 (書き込み系は `N > 0` を前提とする panic を維持し、doc の `# Panics` と `should_panic` で明文化)
- `SlidingWindow::push` / `SimpleMovingAverage::observe` / `RingBuffer::push` の `N = 0` が index out of bounds / 0 除算で落ちていたのを、理由を述べた `assert!` に変更 (前提違反であることが message から分かる、挙動は panic のまま)
- `stats::StreamingStats::skewness` の `m2.powf(1.5)` を `m2 * m2.sqrt()` に (数学的に同一、両方 IEEE 正確丸めなので決定論かつ `powf` より誤差が小さい)

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
