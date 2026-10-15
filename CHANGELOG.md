# Changelog

All notable changes to ALICE-Analytics will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- CI (G11、検査器自身の試験): clippy の禁止一覧 (G1 の超越関数、G10 の wall clock / OS 乱数) に canary を追加 `scripts/clippy_canary.py` が repo の `clippy.toml` / `rust-toolchain.toml` / `[lints]` をそのまま写した使い捨て crate に、`scripts/clippy_canary_calls.tsv` の 1 行ごとに禁止呼び出しを 1 つ置き (`f64::sin` / `f64::sin_cos` / `Instant::now` / `SystemTime::now` / `RandomState`)、CI の clippy job と同じ引数で clippy を走らせて、行ごとにその path を名指す `disallowed_*` 診断がちょうど 1 件出ること・禁止呼び出しの無い対照の関数には 1 件も出ないこと・各行の path が `clippy.toml` にあることを確かめる data file が 0 行、または `clippy.toml` が 0 件の時は「比較 0 件」で失敗する 試験 `scripts/test_clippy_canary.py` (cargo 無し) と共に CI の clippy job と preflight で走る `Instant::elapsed` / `SystemTime::elapsed` は現在の `clippy.toml` に無いので行に含めていない
- CI (G11): `scripts/run_tests.sh` (ALICE-DB と byte 一致、変更なし) の試験 `scripts/test_run_tests.py` を追加 cargo の出力を模した log を流し、通常の成功・試験 0 件・指定した file が 0 件または build されていない・似た名前の file・command の失敗・Windows の出力 (backslash・CRLF・ANSI 色) で終了コードと message を確かめる docs-lint job (3 OS) と preflight で走る
- CI (G4 / G6): debug profile の test step (全 feature / 既定 feature) を `scripts/run_tests.sh` 経由にして、`determinism_golden` と `float_edge_golden` が 1 件も走らない時に失敗させる (全 suite は従来どおり走る) `tests/determinism_golden.rs` に `assert_golden` 自身の canary を 3 本追加 (外部で計算した SHA-256 を受け入れる対照、下限より短い出力で panic、異なる hash で panic) 記録済みの golden 値は変えていない
- 試験: `tests/float_edge_golden.rs` (21 件) が符号付きゼロ・近い大きな値の桁落ち (1e8 + {0, 1, 2} 等)・逆符号の和・平均 x の大きい共線点・分散 0 の入力に対する `StreamingStats` / `LinearRegression` / `ExponentialMovingAverage` / `EwmaDetector` / `SlidingWindow` / `quantile_sorted` / `iqr` の出力を `to_bits()` で bit 一致として固定する (許容誤差なし、各値は閉形式と照合してから記録) CI に `Release profile` workflow (ubuntu / macOS) を追加し、`determinism_golden` と `float_edge_golden` を `cargo test --release` で走らせる `scripts/run_tests.sh` (ALICE-DB と byte 一致) が試験 0 件、または指定した file で 0 件の時に失敗させる preflight (full) でも走る

### Changed
- CI: root の Cargo.lock を追跡するようにした golden の試験は解決した alice-det-math の識別子を固定するので、依存の新しい版が出るたびに main の CI が黙って落ちないよう、記録した版で build する 利用者と同じく lock 無しで解決した場合は、Security & Hygiene の package integrity が展開した .crate で lock 付きの check の後に lock を外して `cargo check --all-features` し、alice-det-math が解決されたことを確かめる
- CI: 公開する文面の検査の pin を ALICE-Law f9b9000 (checker sha256 84226ec7…) に進めた 新しい版で HEAD の全 file に指摘 0 件
- CI / `clippy.toml`: 決定論 gate (G10) として `std::time::Instant::now` / `std::time::SystemTime::now` / `std::collections::hash_map::RandomState::new` を disallowed-methods に追加 この crate が計算する推定値は同じ入力ならどの機械でも一致することを意図しており、production code が wall clock や OS randomness を読む経路を持つと同じ入力で別の値になりうる `scripts/entropy_guard.py` (試験 `scripts/test_entropy_guard.py`) が同じ対象を compile 無しで静的に検査し、許容する呼び出し箇所ごとに件数と理由を固定する 現状許容しているのは `privacy.rs` の既に deprecated な `XorShift64::from_entropy` 系列のみ (`SystemTime::now` 1 箇所・`UNIX_EPOCH` 2 箇所・`from_entropy(` 4 箇所、0.5.0 で module ごと削除予定、本 gate はその削除を早めない) docs-lint job (3 OS) と preflight で走る
- CI: build した library と実行形式を tracked にしない検査 (`scripts/tracked_binaries.py`、ALICE-SDF と byte 一致) を追加 例外は `.github/tracked-binaries.txt` に理由付きで書いた path だけ、読めない file と走査 0 件でも失敗する CI (docs-lint job、全 OS) と preflight で走る
- CI: 全 workflow に concurrency group があり、branch の run は新しい更新で打ち切られ main の run は打ち切られないことを `scripts/workflow_concurrency.py` (試験 `scripts/test_workflow_concurrency.py`、ALICE-LOL / ALICE-SDF / ALICE-DetMath と byte 一致) が検査する workflow を 1 つも読めなければ失敗する CI (actionlint job) と preflight で走る
- CI: clippy の禁止一覧に `f32::sin_cos` / `f64::sin_cos` を足した (ALICE-Law G1 の一覧と揃える) 現時点で repo に呼び出しは無い
- CI: 上限の無かった 16 job に `timeout-minutes` を付けた (止まった step は共有 runner を GitHub の上限の 6 時間まで占有しうる) 値は直近の成功した run 10 本の所要時間の最大の 2 倍・中央値の 4 倍の大きい方を 5 分単位に切り上げ、cargo を使う job は最低 15 分 (cache が外れた build のため)、他は最低 10 分 既に上限のある job (fuzz) は変えていない `scripts/workflow_timeouts.py` (試験 `scripts/test_workflow_timeouts.py`、ALICE-LOL と byte 一致) が全 job の上限を検査し、job を 1 つも読めなければ失敗する CI (actionlint job) と preflight で走る
- **Behavior change:** `alice-det-math` を 0.4 から 0.5 に上げた (`law` feature では `alice-zip` を 0.9 から 0.10 に) `alice_analytics::SEMANTICS_ID` (再 export) の値が `d2209b30…` から `135c407d…` に変わる 推定値・分位点・cardinality・noise の出力は変わらない (determinism golden の 6 場面 privacy / stats / streaming_ops / anomaly / window / sketch が 0.5 の上で bit 一致) `law` feature の `ResidualSummary::law_id` / `ResidualSketch::law_id` は識別子を混ぜるので値が変わる (固定の法則で `bd2c0c24…` から `c8bc7356…`、同じ法則を旧識別子で計算すると `bd2c0c24…` に戻るので、変わったのは識別子だけ) 0.4.0 以前に保存した `law_id` は 0.5.0 の値と一致しない `law` feature の公開型は `alice-zip` 0.10 の型になる
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る `.public-text.toml` は文書の linter とその試験が意図して持つ語を、種類ごとの行数で許可する (行数が変われば失敗)
- CI: `ci.yml` / `fuzz.yml` / `security-audit.yml` の `concurrency` を、main では commit ごとの独立した group にして (`cancel-in-progress` を main では false にして) main の run が後続の push で打ち切られないようにした main 以外の branch では従来どおり同じ branch の新しい push が古い run を打ち切る

### Fixed
- CI (`scripts/entropy_guard.py`、G10): 2 つの穴を修正 (1) `RandomState::new` だけを検査しており `RandomState::default()` (同じ OS 乱数シードのハッシャーに届く) を素通りさせていた `RandomState` の裸の語を検査する形に変更 (clippy 側の `disallowed-methods` は `Default::default` をトレイトメソッドとして実装型だけでは指せないため、この script が唯一の検査層だったので `disallowed-types` で型そのものを ban する層も追加、`HashMap::new` 内部の無名使用は非該当のまま確認済) (2) `#[cfg(test)]` module 判定のブレース深さ計算が文字列しか除去しておらず、`'{'` のような char literal がブレース深さを狂わせて test module 判定が早期終了し、直後の production code を見逃す経路があった 文字列・raw string・char literal・ブロックコメントを 1 pass で正しく除去する sanitizer に置換 (3) `<std::time::Instant>::now()` のような qualified-self 呼び出しが `Instant::now` のパターンに `>` を挟むため素通りしていた (`as Trait` を挟む形も含む) 型名と `::method` の間に `>` / `as Trait` を許容するよう正規表現を拡張 canary test 6 本追加 (元の script に対して赤を確認) 比較件数は変わらず 11 file / 10 pattern / 0 problems
- **Behavior change:** `StreamingStats::observe`、`SlidingWindow::min`/`max`、`DDSketch` の `insert` と `merge` (最小値・最大値の更新) を `<`/`>` 比較から `alice_det_math::min64`/`max64` に変えた IEEE 754 では `-0.0 == +0.0` なので、走行中の最小値・最大値がどちらかの符号のゼロを持つと `<`/`>` は発火せず、最初に来たゼロの符号に固定されていた (arch に依らず、同じ多重集合でも観測順で符号が変わる) 実測 (17 入力 × 8 出力 (stats / window / sketch insert / sketch merge の min / max) を aarch64 / x86_64 × debug / release で `to_bits` 比較): 修正前も後も 4 構成の全出力が一致し、修正で変わったのは混在ゼロを含む 5 入力の 20 出力 (各構成で同じ、うち sketch が 10) だけで、非ゼロ / +0 のみ / -0 のみ / NaN / 大きな値 / 200 点の固定入力は不変 試験 `tests/minmax_signed_zero_order.rs` (両方の順序、sketch は insert と merge) 各 file の 1 箇所を `f64::min`/`max` に戻すと x86_64 で red (aarch64 は IEEE どおりなので green)、修正前の `<`/`>` に戻すと両 arch で red
- `alice-det-math` の要求を 0.5.1 に上げた (`min64`/`max64` は 0.5.1 で追加) 0.5.1 で `SEMANTICS_ID` が `135c407d…` から `85e08846…` に変わるので、`tests/determinism_golden.rs` の `GOLDEN_SEMANTICS_ID` と `GOLDEN_LAW_ID` (`c8bc7356…` から `c32f77f9…`) を記録し直した 動いたのは識別子の入力だけで、0.5.0 の識別子 (0.5.1 の `PREVIOUS_SEMANTICS_IDS` にある) を渡すと law_id は以前の `c8bc7356…` のまま (同じ file の試験で固定)、他の golden は不変

## [0.4.0] - 2026-10-11

### Deprecated
- `privacy` の全ての公開 item (`XorShift64` / `LaplaceNoise` / `RandomizedResponse` / `Rappor` / `RAPPOR_BITS` / `PrivacyBudget` / `PrivateAggregator`) を deprecated にした (0.5.0 で削除) **差分プライバシーになっていない**ため: noise 源の `XorShift64` は出力がそのまま内部状態で、生の `next_u64` なら 1 回、noise の標本 (上位 53 bit が出る) なら 1〜2 回の観測で以降が全て決まる / `from_entropy` は時刻から seed を作る / `LaplaceNoise::sample` の浮動小数点の逆関数法は下位 bit から一様乱数が漏れる (Mironov 2012) 代わりは `alice_crypto::dp` (鍵つき ChaCha20、離散 Laplace、定数時間の標本化、`dp_int` / `dp_sum` / `randomized_response` / `bernoulli_ratio`、0.4.0) この crate は差分プライバシーの機構を持たない方針で、写しも依存もしない module の doc に Security の節を置いた ALICE-* の他 repo に利用者は無い (手元の全 clone と GitHub の code search で確認) 削除までの間は `scripts/deprecation_pin.py` (CI の全 OS と preflight) が固定する: module の公開 item に `#[deprecated]` が付いていること (比較 0 件は失敗) / `allow(deprecated)` は module 自身と値を固定している 3 つの test file だけに、理由の comment を直前に置いて許す

### Fixed
- `scripts/docs_lint.py` が空の `[Unreleased]` (release を切った直後の状態) を「比較 0 件」として失敗にしていた 空の `[Unreleased]` (中身の無い分類見出しだけの場合を含む) は `Cargo.toml` の版にその版の節がある時だけ通し、版を上げたのに節が無い時は失敗にする 本文があるのに分類見出しが無い時は従来どおり失敗
- `cargo clippy --all-targets --no-default-features` が試験の compile で失敗していた (CI はこの組み合わせを走らせていなかった) std 専用の `RandomizedResponse::new` を呼ぶ試験 (`golden_privacy` と `panic_contract` の 5 本) に `#[cfg(feature = "std")]` を付けた no_std の単体試験の build では libtest が std を link し inherent method が優先されるので、`FloatExt` (libm への委譲) は `not(any(feature = "std", test))` に限った 委譲するのは正確丸めの演算 (`mul_add` / `sqrt` / `ceil` / `floor` / `round`) だけなので結果の bit は変わらない CI の clippy job と preflight にこの組み合わせを足した
- Fuzz の workflow が crash を見つけても成功していた (run の step が `continue-on-error`) crash で job を失敗させ、各 target が 1 件以上の入力を実行したことを確かめ (0 件は失敗)、target ごとの実行数と coverage を job summary に出す `fuzz/regressions/<target>` の入力を毎回 corpus として再生する
- `fuzz_metric_aggregate` が試験の側で panic していた (crate の欠陥ではない): sorted を前提とする API に渡す配列を `partial_cmp().unwrap_or(Equal)` で並べており、NaN を含むと全順序にならず標準の sort が「total order を実装していない」で panic する `f64::total_cmp` に替え、crash の入力を `fuzz/regressions/fuzz_metric_aggregate/` に置いた (修正後 846 万件で crash 無し) crate の `src/` に同じ比較は無い

### Changed
- **license:** 0.4.0 から Apache-2.0 だけにする (`Cargo.toml` の `license = "Apache-2.0"`、`LICENSE-MIT` を削除、`NOTICE` を追加) 0.3.0 までの公開済の版は MIT OR Apache-2.0 のまま 再配布には `LICENSE-APACHE` と `NOTICE` を添える `scripts/license_check.py` (CI の全 OS と preflight) が manifest・file・README・`NOTICE` の文言が揃っていることを確かめ (比較 0 件は失敗)、CI は package に `LICENSE-APACHE` と `NOTICE` が入ることを確かめる この crate に依存する `alice-physics` / `alice-db` (どちらも AGPL-3.0-or-later OR 商用) と `alice-edge` (MIT OR Apache-2.0) は、いずれも Apache-2.0 の crate に依存できる (permissive な依存のまま)
- **破壊的変更 (`law` feature):** `alice-zip` を 0.7 から 0.9 に上げた `law::ResidualSummary` / `ResidualSketch` / `residual_summary` が受け渡す `alice_zip::law::SignalLaw` は依存先の版ごとに別の型なので、`law` を使う利用者が自分で `alice-zip` 0.7 / 0.8 の `SignalLaw` を作って渡している場合は compile できなくなる (利用者も `alice-zip` 0.9 に上げる) 値の意味は変わらない: `law_id` と推定値の digest は `tests/determinism_golden.rs` (9 件) / `tests/law_identity.rs` (8 件) の記録値のまま 0.9.0 で通る 0.8 / 0.9 の変更 (残差 container の codec の選択、container 形式) はこの crate が使う `SignalLaw` の評価と `law_id` に触れない `cargo semver-checks` はこの型の変化を検出しない (0.3.0 との比較を minor として判定させても 196 件すべて通る) ので、版は公開 API の型から決めた

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
