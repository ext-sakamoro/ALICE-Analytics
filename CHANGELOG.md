# Changelog

All notable changes to ALICE-Analytics will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Deprecated
- `privacy` の全ての公開 item (`XorShift64` / `LaplaceNoise` / `RandomizedResponse` / `Rappor` / `RAPPOR_BITS` / `PrivacyBudget` / `PrivateAggregator`) を deprecated にした (0.5.0 で削除) **差分プライバシーになっていない**ため: noise 源の `XorShift64` は出力がそのまま内部状態で 1 回の出力から以降が全て決まる / `from_entropy` は時刻から seed を作る / `LaplaceNoise::sample` の浮動小数点の逆関数法は下位 bit から一様乱数が漏れる (Mironov 2012) 代わりは `alice_crypto::dp` (鍵つき ChaCha20、離散 Laplace、定数時間の標本化、`dp_int` / `dp_sum` / `randomized_response` / `bernoulli_ratio`、0.4.0) この crate は差分プライバシーの機構を持たない方針で、写しも依存もしない module の doc に Security の節を置いた ALICE-* の他 repo に利用者は無い (手元の全 clone と GitHub の code search で確認)

### Fixed
- Fuzz の workflow が crash を見つけても成功していた (run の step が `continue-on-error`) crash で job を失敗させ、各 target が 1 件以上の入力を実行したことを確かめ (0 件は失敗)、target ごとの実行数と coverage を job summary に出す `fuzz/regressions/<target>` の入力を毎回 corpus として再生する
- `fuzz_metric_aggregate` が試験の側で panic していた (crate の欠陥ではない): sorted を前提とする API に渡す配列を `partial_cmp().unwrap_or(Equal)` で並べており、NaN を含むと全順序にならず標準の sort が「total order を実装していない」で panic する `f64::total_cmp` に替え、crash の入力を `fuzz/regressions/fuzz_metric_aggregate/` に置いた (修正後 846 万件で crash 無し) crate の `src/` に同じ比較は無い

### Changed
- **破壊的変更 (`law` feature):** `alice-zip` を 0.7 から 0.8 に上げた `law::ResidualSummary` / `ResidualSketch` が受け渡す `alice_zip::law::SignalLaw` は依存先の版ごとに別の型なので、`law` を使う利用者が自分で `alice-zip` 0.7 の `SignalLaw` を作って渡している場合は compile できなくなる (利用者も `alice-zip` 0.8 に上げる) 値の意味は変わらない: `law_id` と推定値の digest は `tests/determinism_golden.rs` / `tests/law_identity.rs` の記録値のまま通り、0.8 の追加は残差 container の codec の選択だけ この変更のため次の版は 0.4.0 (0.x の minor) になり、`privacy` の非推奨もその版に入る

### Removed
- `src/db_bridge.rs` / `src/queue_bridge.rs` / `src/python.rs` / `pyproject.toml`: `lib.rs` から参照されず、要求する依存 (`alice-db` / `alice-queue` / `pyo3` / `numpy`) も `Cargo.toml` に無いためコンパイルされない 4 file 公開パッケージには同梱されていたが機能はしていなかった `pyproject.toml` は存在しない `pyo3` feature を指定しており、`pyproject.toml` と `queue_bridge.rs` の license 表記 (AGPL-3.0) は crate 本体 (`MIT OR Apache-2.0`) と矛盾していた Python バインディングと各ブリッジを再開する場合は、依存と feature と license 表記を揃えた上で改めて追加する (内容は履歴に残る)

## [0.3.0] - 2026-10-08

### Added
- `alice_analytics::SEMANTICS_ID`: この crate の推定値が計算される算術の識別子 (`alice-det-math` が自身の数値的な振る舞いに付ける 32 byte の定数を crate root から再 export) 保存・転送・merge される推定値の隣に記録すると、2 つの結果が数として比較可能かを後から言える
- `law::ResidualSummary::law_id` / `law::ResidualSketch::law_id`: 要約がどの `f(x)` のものか、どの算術で計算されたかを表す識別子 (`alice_zip::law::SignalLaw::law_id` を `SEMANTICS_ID` の下で取ったもの) 法則の定義域と係数と算術を覆い、法則の根拠・残差・出所は覆わないので、別の測定から得た同じ `f(x)` は同じ識別子を共有する

### Changed
- **Breaking:** `law::ResidualSummary` に `law_id` field が増え、`#[non_exhaustive]` が付いた crate 外から struct literal で構築できなくなる (本型は要約の生成側が作って読み手が受け取る型で、構築経路は `ResidualSketch::summary` / `residual_summary` のみ)
- `alice-det-math` を 0.3 から 0.4 に、`alice-zip` を 0.5.1 から 0.7 に上げた 0.4 以降の `alice-det-math` は算術の識別子を持つ (0.3 には無く、0.3 を解決する消費者と同じ依存グラフでは「どの算術で出た数か」を言えない) `ln` / `exp` / `powf` の出力と `SignalLaw` の fit / evaluate は版の間で bit 不変で、既存の決定論ハッシュ 7 本は変わっていない
- semver-checks の CI job を informational から gate にした 宣言した版の上げ幅が API の変更を覆うことを確認する pass と、lint を強制して比較件数が 0 でないことを確認する pass の 2 本 (上げ幅が最大の時は全 lint が skip され、何も比較しないまま exit 0 になるため) `scripts/preflight.sh` にも同じ 2 本を追加

## [0.2.0] - 2026-10-07

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

- `DDSketch*::non_finite()`: 非有限として弾いた sample の数 (`law::ResidualSummary::non_finite` と同じ分類を同じ名前で数える) 公開 API の追加のみで、既存 accessor の signature は変えていない
### Changed
- `rust-version = "1.87"` を宣言 (1.86 では `is_multiple_of` が未安定でビルドできないことを実測)
- CI: test を 4 OS (macOS / Linux x86_64 / Linux arm64 / Windows) に、`law` 有りの `no_std` ビルド、MSRV、rustdoc (既定 / 全 feature)、docs lint (3 OS)、example 実行を追加 security-audit は cargo audit の DB を `target/` 配下に置き、install を `taiki-e/install-action` に変更、`scripts/preflight.sh` を CI と同じ引数に揃え `--quick` で `cargo test --lib` を実行
- 何もしない composite action を削除、deny.toml の license 許可を依存グラフに存在するもの (MIT / Apache-2.0) に限定し wildcard と未知の registry / git を deny
- README を全面改稿 (ライセンス表記を Cargo.toml の `MIT OR Apache-2.0` に一致させ、存在しない bridge module の記述を削除)
- `DDSketch` の範囲外の値に関するコメントの上端を `accurate_range()` に合わせて訂正
- `Cargo.toml`: `std` feature が `alice-det-math/std` を伝播する `[lints.clippy]` に `suboptimal_flops` / `imprecise_flops` の `allow` を理由付きで追加 (det-math 経由の呼出を `mul_add` / platform `libm` の形に書き戻す提案を crate の性質として 1 箇所で止める、`alice-det-math` 自身と同じ形) `[dev-dependencies]` に `sha2` (golden ハッシュ用)
- `tests/analytic_oracle.rs` / `tests/law_residual.rs` / `examples/residual_summary.rs`: 禁止した inherent メソッド (`powf` / `exp` / `powi`) の呼出を `alice_det_math` 経由と結合順固定の整数冪に置換 期待値は変わらない (`ipow64` は `powi` と bit 一致、実測)
- README の `no_std` の浮動小数点に関する記述を訂正 (超越関数は `std` の有無に関わらず `alice-det-math` 経由で、`libm` を使うのは `sqrt` / 丸め / `mul_add` だけ) 関連 crate に `alice-det-math` を追加

- CI の `test` job と `scripts/preflight.sh` に `cargo test --lib --no-default-features` と `--features law` の 2 lane を追加 `no_std` の unit test が host で走る唯一の経路で、`libm` の丸め path (`sqrt` / `ceil` / `floor` / `round` / `mul_add`) をここで実行する `--lib` のみなので 1 OS あたり数秒
- `RandomizedResponse::new` の `p_true` の式を `e^ε/(1 + e^ε)` から数値安定形の `1/(1 + e^(−ε))` に変更 旧式は `e^ε` が `ε ≳ 709.79` で `inf` になり `inf/inf = NaN` を返していた 上端を assert で切るのでなく式を直したのは、**NaN が代数的な書き方の副産物で定義域の問題ではない**ため (`ε → ∞` の極限 `1.0` は意味のある設定) 実測: `ε = 0 / 0.5 / 1 / 50 / 709` では旧式と bit 一致、`ε = 3` で 1 ulp 異なる ⇒ `tests/determinism_golden.rs` の `GOLDEN_PRIVACY` を更新した
- `tests/determinism_golden.rs` の `GOLDEN_SKETCH` を更新 (`sketch` シナリオに非有限 sample の投入と `non_finite()` / `sum()` / `min()` / `max()` の直列化を追加したため)
### Fixed
- **`DDSketch::insert` が非有限 / 極大の大きさで bucket index を overflow させていた** — `bucket_index` の `f64_i32(...) + self.offset` は、`value` が `inf` (上流の 0 除算が届いた場合) や `alpha = 0` (γ = 1 ⇒ `inv_ln_gamma` が無限大) のとき `as i32` が `i32::MAX` に飽和した上で offset を足すため i32 を溢れる debug ビルドでは panic し、**release ビルドでは負に wrap して直後の `max(0)` で bin 0 に入っていた** (= 2026-09-17 に修正した「範囲外の値が rank をずらす」と同型の silent な破損) `saturating_add` にして、doc の既存契約どおり上端 / 下端の edge bin に収容する
- **`TumblingWindow` / `HierarchicalRollup` が `u64::MAX` 近傍の時刻で overflow していた** — `current_start + window_ms` が debug では panic、release では wrap して `end_ms < start_ms` の結果を出していた `saturating_add` に変更 イベント数の計上は変わらない
- `RingBuffer::capacity()` が `N = 0` で underflow、`is_full()` が `% N` で 0 除算していた `N.saturating_sub(1)` と `N <= 1` の早期 return にして read 系アクセサを全域化 (書き込み系は `N > 0` を前提とする panic を維持し、doc の `# Panics` と `should_panic` で明文化)
- `SlidingWindow::push` / `SimpleMovingAverage::observe` / `RingBuffer::push` の `N = 0` が index out of bounds / 0 除算で落ちていたのを、理由を述べた `assert!` に変更 (前提違反であることが message から分かる、挙動は panic のまま)
- `stats::StreamingStats::skewness` の `m2.powf(1.5)` を `m2 * m2.sqrt()` に (数学的に同一、両方 IEEE 正確丸めなので決定論かつ `powf` より誤差が小さい)

- **`PrivacyBudget::try_spend` が負の ε を受け付け、残予算を増やしていた** — 判定が予算比較 `total + epsilon <= max` だけだったので負値は無条件に通り、`try_spend(-10.0)` が `true` を返して残予算が 1 → 11 に、`-inf` では無限大になっていた 正直な 2 回の問い合わせの間に負の ε を挟めば、この型が存在する理由である上限を回避できる 非有限と負値を明示的に拒否し、拒否時は `total_epsilon` / `query_count` を一切触らない (`NaN` / `+inf` は従来も拒否されていたが、比較が偽になる副作用としてだった) `0.0` / `-0.0` は従来どおり well-formed (課金 0、問い合わせ 1 件として計上)
- **`RandomizedResponse::with_probability` に `NaN` を渡すと `p_true()` が `NaN` を返していた** — `clamp(0.5, 1.0)` は self が `NaN` のとき `NaN` を返すので丸められずに残っていた 定義域 `[0.5, 1.0]` の扱いを同 crate の `ExponentialMovingAverage::new` に揃え、範囲外は黙って丸めず panic する (`# Panics` に記載) **Breaking:** `0.0` → `0.5`、`2.0` → `1.0` の暗黙の丸めも無くなる 丸めは呼び出し側が頼んだのと違う ε の機構を作るので、推定値の意味が変わる `0.5` (常に無作為) と `1.0` (常に正直) は意味のある端なので許す
- **`EwmaDetector::new` / `set_alpha` に `NaN` を渡すと検出器が黙って無効化されていた** — `alpha.clamp(0.001, 1.0)` は self が `NaN` のとき `NaN` を返すので丸められずに残り、以後 `ewma` / `std_dev` / `anomaly_score` がすべて `NaN` になる `NaN` との比較は全て偽なので `is_anomaly` が**常に `false`** を返す (無限大の score は「異常」と読めるが `false` は「正常」と読めるので害が大きい) 定義域 `(0.0, 1.0]` の扱いを `ExponentialMovingAverage::new` に揃えて panic にした setter にも同じ検査を置く (丸めたままだと構築時の検査を後から回避できる) `CompositeDetector::with_thresholds` は `ewma_alpha` をそのまま渡すので同じ契約を継承する (`# Panics` に記載) **Breaking:** `0.0` と負値が `0.001` に、`1.0` 超が `1.0` に黙って丸められていた挙動も無くなる (実測で `-1.0` と `0.0` は同一結果になっていた) 分散推定 0 のときの `anomaly_score` が `inf` を返すことは契約として doc に明記し値で固定した (`alpha = 1.0` 固有ではなく、`alpha = 0.3` の定数 stream でも同じ状態になる)
- **`RandomizedResponse::new` が定義域を検査していなかった** — `with_probability` だけが検査していたので、同じ不変条件を `new` 経由で破れた 実測: `epsilon = -5.0` で `p_true = 6.69e-3` (0.5 未満 = `estimate_proportion` の推定値の符号が反転する領域)、`epsilon ≳ 709.79` / `NaN` / `inf` で `p_true = NaN` 有限かつ非負の `epsilon` を要求する (検査する量は入力の `epsilon` で、`p_true` は導出値) **Breaking:** `new(-5.0)` は値を返さず panic する `epsilon = 0.0` は許す (`p_true = 0.5` = 常に無作為、`with_probability(0.5)` と同じ端)
- **`DDSketch::insert` が `NaN` を「ちょうど 0 の sample」として分類していた** — `NaN` は `value > 0.0` も `value < 0.0` も偽になるので `zero_count` に入り、`count` に数えられ `sum` (したがって `mean`) を汚し、全 `NaN` の stream が `quantile(q) == 0.0` を返していた 非有限 (`NaN` / `±inf`) は大きさではないので、同 crate の `law::PointClass::NonFinite` (「NaN または無限大」を 1 つの類として扱い要約から外す) に揃えて専用の counter に数え、`count` / `sum` / `min` / `max` と bin には一切触れない したがって quantile の裏にある rank は実際に bin に入った値の rank になる `merge` は counter を加算し `clear` は 0 に戻す **Breaking:** 観測可能な挙動が 4 つ変わる (1) `count()` が非有限を数えない (2) `sum()` / `mean()` が `NaN` に汚染されない (3) `min()` / `max()` が `±inf` で広がらない (4) `±inf` が端の bin に入らなくなる (2026-10-07 の前半で `saturating_add` の契約として doc に書いた「`±inf` も端の bin」は、この前例合わせで撤回する なお `saturating_add` 自身は**有限**の極大値 + 小さい `alpha` で到達するので引き続き必要で、`tests/panic_contract.rs` に有限入力の oracle を置いた)
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
