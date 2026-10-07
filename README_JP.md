# ALICE-Analytics

[English](README.md)

誤差の上限が明示された確率的データ構造とストリーミング統計の crate HyperLogLog
による異なり数推定、相対誤差を保証する DDSketch の分位点、Count-Min による頻度と
heavy hitter、ストリーミングのモーメント / 共分散 / 回帰、ウィンドウ集計、異常検知、
局所差分プライバシーを含む コアは `no_std` で、sketch は固定長でスタックに置かれる
`law` feature を有効にすると、点の集合が `alice_zip::law::SignalLaw` からどれだけ
ずれているかの分布も要約できる

時系列データベース、メトリクスのバックエンド、ダッシュボードではない 要約をメモリ上に
持つだけで、保存と転送は呼び出し側に任せる sketch の答えは近似であり、厳密な分位点や
厳密な異なり数が必要なら生の値を保持する

License: MIT OR Apache-2.0

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
- [Law の残差分布](#law-の残差分布)
- [Feature](#feature)
- [モジュール](#モジュール)
- [誤差の上限](#誤差の上限)
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

`std` 無しでは浮動小数点関数は `libm` を使い (プラットフォームのライブラリと最後の
ulp で異なりうる)、プライバシー機構は seed を明示して作る CI は
`thumbv7em-none-eabihf` 向けに `law` 有り・無しの両方でライブラリをビルドする

## モジュール

| モジュール | 内容 |
|-----------|------|
| `sketch` | `HyperLogLog10/12/14/16`、`DDSketch128` … `DDSketch2048`、`CountMinSketch1024x5` / `2048x7` / `4096x5`、`HeavyHitters5/10/20`、`FnvHasher`、`Mergeable` |
| `stats` | `StreamingStats` (平均 / 分散 / 歪度 / 尖度)、`CovarianceMatrix`、`quantile_sorted`、`percentile_rank`、`iqr` |
| `window` | `TumblingWindow`、`SlidingWindow`、`HierarchicalRollup` |
| `streaming_ops` | `SimpleMovingAverage`、`ExponentialMovingAverage`、`ChangeRate`、`LinearRegression`、`LinearRegressionFull` |
| `anomaly` | `StreamingMedian`、`MadDetector`、`ZScoreDetector`、`EwmaDetector`、`CompositeDetector` |
| `privacy` | `LaplaceNoise`、`RandomizedResponse`、`Rappor`、`PrivacyBudget`、`PrivateAggregator`、`XorShift64` |
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

## 最小サポート Rust バージョン

`rust-version = "1.87"` (実測: 1.86 ではライブラリがビルドできない) CI は 1.87 で
既定 feature と全 feature のライブラリを検査する 開発用ツールチェーンは
`rust-toolchain.toml` で固定している

## ビルドとテスト

```sh
cargo test --all-features
cargo build --lib --no-default-features --features law --target thumbv7em-none-eabihf
scripts/preflight.sh          # every CI gate that runs locally
scripts/preflight.sh --quick  # static checks, clippy, builds, docs, cargo test --lib
```

## 関連 crate

- [`alice-zip`](https://crates.io/crates/alice-zip) — `law::SignalLaw`: 成立範囲、
  残差統計、出典、oracle ケースを伴う fit 済みの law `law` feature はその残差分布を
  要約する

## ライセンス

[Apache License, Version 2.0](LICENSE-APACHE) または [MIT license](LICENSE-MIT)
のいずれかを選択して利用できる
