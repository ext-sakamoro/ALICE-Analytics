# ALICE-Analytics

[English](README.md)

誤差の上限が明示された確率的データ構造とストリーミング統計の crate HyperLogLog
による異なり数推定、相対誤差を保証する DDSketch の分位点、Count-Min による頻度と
heavy hitter、ストリーミングのモーメント / 共分散 / 回帰、ウィンドウ集計、異常検知を
含む コアは `no_std` で、sketch は固定長でスタックに置かれる
`law` feature を有効にすると、点の集合が `alice_zip::law::SignalLaw` からどれだけ
ずれているかの分布も要約できる

時系列データベース、メトリクスのバックエンド、ダッシュボードではない 要約をメモリ上に
持つだけで、保存と転送は呼び出し側に任せる sketch の答えは近似であり、厳密な分位点や
厳密な異なり数が必要なら生の値を保持する

License: Apache-2.0 (0.3.0 以前は MIT OR Apache-2.0)

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
- [Law の残差分布](#law-の残差分布)
- [Feature](#feature)
- [モジュール](#モジュール)
- [誤差の上限](#誤差の上限)
- [決定論](#決定論)
- [最小サポート Rust バージョン](#最小サポート-rust-バージョン)
- [ビルドとテスト](#ビルドとテスト)
- [関連 crate](#関連-crate)
- [ライセンス](#ライセンス)

## インストール

```sh
cargo add alice-analytics
# with the residual summary of alice-zip laws
cargo add alice-analytics --features law
```

## 使用例

```rust
use alice_analytics::sketch::{DDSketch, HyperLogLog};

// quantiles with a relative-error guarantee of 1 %
let mut latency = DDSketch::new(0.01);
for ms in 1..=1000 {
    latency.insert(f64::from(ms));
}
let p99 = latency.quantile(0.99);
assert!((p99 - 990.0).abs() <= 0.01 * 990.0);

// distinct count in 16 KiB of registers
let mut users = HyperLogLog::new();
for id in 0..10_000_u64 {
    users.insert(&id);
}
let n = users.cardinality();
assert!((n - 10_000.0).abs() < 0.05 * 10_000.0);
```

## Law の残差分布

`law::residual_summary(&law, &points)` (feature `law`) は、`points` の
`alice_zip::law::SignalLaw` に対する残差 `y − f(x)` を `DDSketch2048` に流し込み、
`ResidualSummary` を返す

| フィールド | 意味 |
|-----------|------|
| `count` | 要約した点の数 |
| `out_of_range` | `x` が law の成立範囲外 (または非有限) の点の数 law はそこで評価せず、外挿もしない |
| `non_finite` | 範囲内だが `y − f(x)` が NaN / 無限大の点の数 数えるだけで要約には入れない |
| `distribution` | 要約した点が無ければ `None` それ以外は `mean`、`min`、`max`、`max_abs` (厳密値) と `p50`、`p90`、`p99` (sketch による近似) |
| `relative_accuracy` | `α`: 報告される残差 `q` の分位点 `q̂` は `\|q̂ − q\| ≤ α·\|q\|` を満たす |
| `accurate_range` | この上限が成り立つ残差の大きさの範囲 0 は常に厳密 |
| `outside_accuracy` | `accurate_range` の外にある 0 でない残差の数 (0 ならすべての分位点で上限が成り立つ) |

報告する分位点は順位 `⌈q·n⌉` の順序統計量で、sketch の定義と同じ 残差は 2 の冪の
スケールで割ってから挿入する (既定は law 自身の残差 RMS 以下で最大の 2 の冪) ので、
丸めを入れずに `accurate_range` を残差の大きさに合わせられる `law::ResidualSketch`
はストリーミング版で、1 点ずつ `push` し、任意の `q` で `quantile(q)` を読み、`α` と
スケールを選べる

```rust,ignore
use alice_analytics::law::residual_summary;

let s = residual_summary(&law, &new_points);
println!("{} summarised, {} outside the valid range", s.count, s.out_of_range);
if let Some(d) = s.distribution {
    println!("p50 {:+e}  p99 {:+e}  max |r| {:e}", d.p50, d.p99, d.max_abs);
}
```

動くプログラムは `examples/residual_summary.rs` にある
(`cargo run --example residual_summary --features law`)

## Feature

| Feature | 既定 | 説明 |
|---------|------|------|
| `std` | yes | 標準ライブラリの浮動小数点関数、エントロピー由来の seed を使うプライバシー機構のコンストラクタ、JSON / Prometheus 出力 |
| `law` | no | `law` モジュール: `alice_zip::law::SignalLaw` の残差要約 (`std` feature 無しの `alice-zip` と `alloc` を使う) |
| `simd` | no | 効果なし 既存の feature 指定がそのままビルドできるよう残している |

`std` 無しではプライバシー機構は seed を明示して作り、`sqrt` / `ceil` / `floor` /
`round` / `mul_add` は `libm` を使う IEEE 754 はこの 5 つに正確丸めを要求するので、
inherent method と同じ bit を返す 超越関数は `std` の有無に関わらず
`alice-det-math` を経由するので、`no_std` ビルドと `std` ビルドは bit まで一致する
([決定論](#決定論) 参照) CI は `thumbv7em-none-eabihf` 向けに `law` 有り・無しの
両方でライブラリをビルドする

## モジュール

| モジュール | 内容 |
|-----------|------|
| `sketch` | `HyperLogLog10/12/14/16`、`DDSketch128` … `DDSketch2048`、`CountMinSketch1024x5` / `2048x7` / `4096x5`、`HeavyHitters5/10/20`、`FnvHasher`、`Mergeable` |
| `stats` | `StreamingStats` (平均 / 分散 / 歪度 / 尖度)、`CovarianceMatrix`、`quantile_sorted`、`percentile_rank`、`iqr` |
| `window` | `TumblingWindow`、`SlidingWindow`、`HierarchicalRollup` |
| `streaming_ops` | `SimpleMovingAverage`、`ExponentialMovingAverage`、`ChangeRate`、`LinearRegression`、`LinearRegressionFull` |
| `anomaly` | `StreamingMedian`、`MadDetector`、`ZScoreDetector`、`EwmaDetector`、`CompositeDetector` |
| `privacy` | **deprecated、差分プライバシーになっていない** (noise 源が予測できる、0.5.0 で削除) 代わりは `alice_crypto::dp` |
| `pipeline` | `MetricPipeline`、`MetricRegistry`、`MetricSnapshot`、`RingBuffer` |
| `export` | `MetricSnapshot` の JSON / Prometheus テキスト出力 (`std`) |
| `law` | `residual_summary`、`ResidualSketch`、`ResidualSummary` (feature `law`) |

## 誤差の上限

| 構造 | 上限 | 検査している test |
|------|------|------------------|
| `DDSketch` | `accurate_range()` 内の大きさで `\|q̂ − q\| ≤ α·\|q\|` (負の値を含む) | `tests/analytic_oracle.rs` |
| `HyperLogLog` | 標準誤差 `1.04/√m` (`m` はレジスタ数) | `tests/analytic_oracle.rs` |
| `CountMinSketch` | 推定値 ≥ 真の頻度、過大分 ≤ `ε·N`、`ε = e/w` | `tests/analytic_oracle.rs` |
| `law::ResidualSketch` | 残差に対する `DDSketch` の上限、`mean` / `min` / `max` は厳密 | `tests/law_residual.rs` |

これらの test の期待値は test ファイル内に書いた閉形式または 2-pass の参照計算で、
検査対象の関数の出力ではない

**非有限の sample** (`NaN` / `±inf`) は大きさではない `DDSketch::insert` はこれを
`non_finite()` に数え、`count` / `sum` / `min` / `max` と bin には触れない
したがって quantile の裏にある rank は、実際に bin に入った値の rank である
`law` モジュールが非有限の残差に対して行う分類と同じ **有限**の大きさが
`accurate_range()` の外にある場合は、数えて最も近い端の bin に入れる
rank は保たれるが `α` の上限は成り立たない

## 決定論

同じ入力は、サポートする全ターゲットで **同じ bit** を返す sketch は複数のマシンで
merge され、テレメトリは再生され、監査は生イベントから分位点を再計算する —
2 つのホストで最後の ulp が違えば、推定値が 2 つに割れる

| 層 | 対象 | bit 一致する理由 |
|----|------|-----------------|
| IEEE の基本演算 | `+ − × ÷`、`sqrt`、`mul_add`、`ceil` / `floor` / `round` | IEEE 754 が正確丸めを要求するので全ターゲットが一致する `mul_add` は単一丸めの融合積和で、FMA 命令の無いターゲットでは正確丸めの software `fma` になる |
| 超越関数 | `ln`、`exp`、`powf` | [`alice-det-math`](https://crates.io/crates/alice-det-math) 経由 上の演算だけを固定した評価順で組んだ実装 プラットフォームの `libm` は **使わない** (最後の ulp が macOS / glibc / MSVC / wasm で異なる) |
| 整数冪 | `DDSketch` の bin 配置で使う `γⁿ` | 結合順を固定した繰り返し二乗 `powi` は乗算木の結合順が未規定なので使わない |
| 擬似乱数 | `XorShift64` とそこから seed を取る全機構 | 状態は整数で、seed を与えるコンストラクタは厳密に再生する エントロピー由来のコンストラクタ (`std` のみ) は設計上再現しない |

強制は機械的に 2 層でかける:

* `clippy.toml` が `f32` / `f64` の inherent な超越関数を `disallowed-methods` に
  並べ、CI が `cargo clippy --all-targets --all-features -- -D warnings` を走らせる
  ⇒ `x.ln()` を書き戻すと silent な乖離ではなくコンパイルエラーになる **書いた
  マシン上で捕まえる**のはこの層
* `tests/determinism_golden.rs` が 7 つのシナリオ (浮動小数点演算を持つモジュール
  ごとに 1 つ) の出力を bit 単位で直列化し、SHA-256 を記録済みの定数と突き合わせる
  各シナリオは直列化した byte 数が 0 でないことも assert するので、モジュールを
  実行しなくなったシナリオは空バッファのハッシュで通るのではなく fail する CI は
  macOS `aarch64`、Linux `x86_64`、Linux `aarch64`、Windows `x86_64` で実行する

### 算術に名前を付ける

マシン間の bit 一致は、保存した推定値に必要なものの半分でしかない 残りの半分は
**どの算術で出た数かを言えること** — ある `ln` の実装で計算した分位点と、別の実装で
計算した分位点は、どれだけ近く見えても 2 つの数である

`alice_analytics::SEMANTICS_ID` がその名前で、`alice-det-math` が自身の数値的な
振る舞いに付ける 32 byte の識別子を再 export したもの 保存・転送・merge される
推定値の隣に記録する 2 つの結果が数として比較可能なのは、この識別子が一致する時
だけ `golden_semantics_id` が hex で固定するので、算術の変更がリリースまで
気付かれずに届くことはない

`law` feature では `law::ResidualSummary::law_id` が両方の半分を一度に名指す
要約した法則自身の識別子を `SEMANTICS_ID` の下で取った値で、`f(x)` が変わった時も
算術が変わった時も変化する 覆うのは `f(x)` が読むもの (定義域と係数) だけで、
法則の得られ方は覆わない ⇒ 別の測定から fit した同じ `f(x)` は同じ識別子を共有する

<!-- claim-test: golden_sketch -->
<!-- claim-test: golden_stats -->
<!-- claim-test: golden_window -->
<!-- claim-test: golden_anomaly -->
<!-- claim-test: golden_privacy -->
<!-- claim-test: golden_streaming_ops -->
<!-- claim-test: golden_law -->
<!-- claim-test: golden_semantics_id -->
<!-- claim-test: golden_law_id -->

**決定論は正しさではない** golden ハッシュは今日の挙動を — 誤りを含めて — 固定する
だけで、上の誤差の上限は閉形式との突合で別に検査している 2 つは独立で、両方が必要

**保証の外**: 基本演算で IEEE 754 に従わないターゲット (SSE2 無しの x87 向け
32-bit x86)、fast-math 系のフラグを付けたビルド、エントロピー由来のプライバシー
機構のコンストラクタ 本 crate や `alice-det-math` の **バージョンを跨いだ** 一致も
保証しない (保証するのは、あるバージョンにおけるプラットフォーム間の一致)
`SEMANTICS_ID` が足すのは「算術が変わらない」という約束ではなく、**変わったことを
言える**ようにすることである

## 最小サポート Rust バージョン

`rust-version = "1.87"` (実測: 1.86 ではライブラリがビルドできない) CI は 1.87 で
既定 feature と全 feature のライブラリを検査する 開発用ツールチェーンは
`rust-toolchain.toml` で固定している

## ビルドとテスト

```sh
cargo test --all-features
cargo test --all-features --test determinism_golden  # プラットフォーム間の bit 一致
cargo test --all-features --test panic_contract      # 退化入力の契約
cargo build --lib --no-default-features --features law --target thumbv7em-none-eabihf
scripts/preflight.sh          # every CI gate that runs locally
scripts/preflight.sh --quick  # static checks, clippy, builds, docs, cargo test --lib
```

## 関連 crate

- [`alice-det-math`](https://crates.io/crates/alice-det-math) — 全ての浮動小数点
  経路が通る、プラットフォーム間で bit 一致する `ln` / `exp` / `powf`
- [`alice-zip`](https://crates.io/crates/alice-zip) — `law::SignalLaw`: 成立範囲、
  残差統計、出典、oracle ケースを伴う fit 済みの law `law` feature はその残差分布を
  要約する

<!-- law-integration:begin law=6a0a5587c15205c25647a16a842e1c3b30bb6abc -->
## ALICE-Law との統合

ALICE-Law の `scripts/gen-law-readme-block.py` が ALICE-Law `6a0a5587c152` とこの repository の `alice-law-conformance.toml` から生成する 手で編集しない

| | |
|---|---|
| release train | alice-analytics 0.5.0 (published)、Law への依存: あり |
| 算術 | train の alice-det-math 0.5.0、算術の識別子 `135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000` この crate の要求は `alice-det-math = "0.5.1"` |
| end to end 検査 | 対象 (`alice-analytics = "0.5"`)、digest `966a78784bb4e3341ac6627d519db896f398176b71c2402b188294db6c8e3728`、aarch64 と x86_64 で同じ |

### 適合 (G1-G14)

| gate | 状態 | 注記 |
|---|---|---|
| G1 no-libm-direct | 未達 | clippy.toml bans 24 of the 25 methods the gate lists as banned, through type-resolved disallowed-methods on f32 and f64 (CI: ci.yml clippy job, cargo clippy … |
| G2 ieee-max-min | 未達 | no Law helper for max/min and no default-deny manifest of raw_* exceptions exist. The library calls f64::max / f64::min / clamp directly at src/anomaly.rs:799, … |
| G3 no-outer-fneg-of-fma | 未達 | no checker for an outer negation of an fma result exists in this repository. mul_add is used in src/stats.rs, src/privacy.rs, src/anomaly.rs and … |
| G4 debug-release-arch-goldens | 未達 | the cross-platform golden file is tests/determinism_golden.rs (golden_sketch, golden_stats, golden_window, golden_anomaly, golden_privacy, … |
| G5 release-profile-test-leg | 未達 | no CI step runs the test suite under the release profile: ci.yml:40,43,50,52 are `cargo test` without --release, and the aarch64 legs (macos-latest, … |
| G6 semantics-id | 未達 | src/lib.rs:80 is `pub use alice_det_math::SEMANTICS_ID;`: the crate re-exports its arithmetic crate's identifier and folds none of its own pins. Its own … |
| G7 handshake-carries-id | 未確認 | audit: the crate has no decode / read / from_bytes / open entry point, no extern "C" or no_mangle item, no pyo3 or wasm-bindgen export, no dyn Any or … |
| G8 persisted-semantics-id | 未確認 | undecided: the crate writes no file or database row, but src/export.rs:24 snapshot_to_json, :58 snapshots_to_json, :86 snapshot_to_prometheus and :146 … |
| G9 persisted-meaning-change-is-major | 未確認 | depends on G8 (the persisted identifier whose history G9 audits), which is undecided. For the in-process identifier the history is: SEMANTICS_ID first appears … |
| G10 no-run-varying-artifacts | 未達 | no lint or scan step bans clock, uuid or unseeded-RNG reads in the library (clippy.toml lists only float methods). A lexical search of src, tests, examples and … |
| G11 gate-self-tests | 未達 | the Law gates that are implemented here have no self-test: the G1 clippy lint has no canary that injects a banned call and confirms the lint still fires (a … |
| G12 public-repo-text-rules | 適合 | the CI job fetches ALICE-Law's own scripts/public_text_check.py at the revision named in .github/public-text-pin and refuses it unless its SHA-256 is the one … |
| G13 no-unspecified-order | 未確認 | audit: no HashMap or HashSet in src (the only one in the repository is tests/analytic_oracle.rs:434, in a test), no rayon, no thread spawn and no par_iter in … |
| G14 float-environment-invariants | 未達 | the build configuration sets no fast-math-style flag and no target-cpu=native: there is no .cargo directory and no build.rs, no RUSTFLAGS in any workflow, no … |

### feature と保証の範囲

| feature | 保証 | 注記 |
|---|---|---|
| `law` | 未申告 |  |
| `simd` | 未申告 |  |
| `std` | 未申告 |  |
<!-- law-integration:end -->

## ライセンス

- [Apache License, Version 2.0](LICENSE-APACHE) (0.4.0 から)
- 再配布 (source でも binary でも) には [LICENSE-APACHE](LICENSE-APACHE) と
  [NOTICE](NOTICE) を添える (Apache-2.0 第 4 条) application の画面に表示する必要は無い
- それより前の版は出した時のライセンスのまま: crates.io の `alice-analytics` 0.3.0 までは
  MIT OR Apache-2.0
- 「ALICE」の名前はライセンスによらず [TRADEMARK_NOTICE](TRADEMARK_NOTICE) の対象
