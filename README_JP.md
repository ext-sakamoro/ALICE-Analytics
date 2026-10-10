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

License: MIT OR Apache-2.0

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

## ライセンス

[Apache License, Version 2.0](LICENSE-APACHE) または [MIT license](LICENSE-MIT)
のいずれかを選択して利用できる
