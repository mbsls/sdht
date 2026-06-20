# Soft Data, Hard Times — Results Summary

*Working summary of the US analysis. Status: US investigation concluded; euro-area replication not attempted.*

## Research question

Does soft (survey) data forecast better than traditional hard macro
indicators during "hard times" — the run-up to and duration of crises?
The motivating claim is that hard data, being backward-looking and
revised, loses its edge in volatile periods while surveys keep informing
turning points.

## Design (one paragraph)

Pseudo-real-time forecasts of US unemployment, industrial production,
nonfarm payrolls, and core PCE inflation, 2001–2024, at horizons
h = 1, 3, 6, 12 months. Strict real-time discipline: every revised
series is sliced from ALFRED vintage cubes as of each forecast origin.
Two parallel windows are dictated by data availability — a **long
window (1998+)** whose only vintage-clean soft series is Michigan
consumer sentiment, and a **short window (2016+)** with a richer soft
block (Michigan sentiment, BBD policy-uncertainty, the NY/Philadelphia/
Dallas Fed manufacturing surveys, and — via a mixed-frequency dynamic
factor — the quarterly Senior Loan Officer survey). Soft data enters
either directly or through a single common factor. Forecast accuracy is
compared with Diebold–Mariano and, conditional on the VIX regime,
Giacomini–White tests. "Hard times" are identified three ways:
ex-post VIX bins, a smooth-transition VAR with the VIX as transition
variable, and a Markov-switching model with a *latent* regime.

Models, in order of sophistication: linear VAR → COVID-dummy VAR →
smooth-transition VAR (STVAR) → Markov-switching AR (MS-AR). Soft factor
built two ways: static PCA and a state-space mixed-frequency DFM.

## What we found, model by model

1. **Linear VAR (baseline).** Ex-COVID, adding Michigan sentiment gives
   small real-activity gains (unemployment h=3 ≈ +7%, IP ≈ +3%) and is
   neutral for inflation — none significant. Through COVID, iterated
   long-horizon forecasts diverge; adding soft data makes the divergence
   worse. *Linear models fail in crises — which is the whole motivation
   for regime models.*

2. **COVID-dummy VAR.** Modeling COVID with a dummy + lag interaction
   (rather than dropping it) cuts full-sample hard-only RMSE by 60–83%
   at long horizons. On this honest full sample, Michigan sentiment is a
   wash-to-slightly-negative; no GW/DM rejection.

3. **STVAR (VIX-threshold regime) — headline.** Two findings:
   - **Regime-switching substantially improves *hard-data* forecasts**:
     vs the COVID-dummy linear VAR, hard-only RMSE falls by 12–62%
     (unemployment), 19–31% (industrial production), 5–42% (payrolls).
   - **Conditional on that, soft data mostly does not help and sometimes
     significantly hurts**: the short-window soft factor significantly
     *worsens* unemployment forecasts (DM p ≈ 0.03). Robustness check
     (a) — dropping the COVID dummy so the high-stress regime identifies
     off both 2008 and 2020 — *strengthens* this (p falls to ≈ 0.002),
     ruling out the "the dummy starved the regime" alternative.

4. **MS-AR (latent regime) — robustness (b).** The latent regime tracks
   the VIX (corr ≈ 0.37) despite never seeing it, so the identification
   is sound. This produced the project's **only hypothesis-consistent
   result**: with the static-PCA factor, soft data significantly helps
   *industrial production* (+5.8%/+8.4% at h=3/6, DM p ≈ 0.03–0.05), and
   the gain is *larger in stress than calm* (h=6: +19.7% stress vs +6.8%
   calm). But it is one target of four (unemployment, payrolls, core PCE
   favor hard; core PCE significantly), and rests on ~13 stress origins.

5. **Mixed-frequency DFM factor (+SLOOS).** A better-constructed soft
   factor — broader (adds credit standards), smoother, properly
   mixed-frequency — moves the result *toward neutrality*: the IP gain
   shrinks to +2–3% and loses significance (p ≈ 0.24), while the harmful
   core-PCE result disappears. With a cleaner factor, soft data is
   approximately a wash.

## Two headline conclusions

**(A) The robust, positive finding is about *hard* data.** Allowing
hard-data dynamics to switch with the volatility regime delivers large,
significant forecast-accuracy gains in and around crises. What is often
described as a "soft-data premium in hard times" appears, in this
design, to be largely a **regime-dependent-dynamics premium that hard
data supplies on its own once modeled nonlinearly.**

**(B) We find no robust evidence that soft data adds incremental value
in volatile periods.** Across model classes and factor constructions,
the soft block ranges from mildly harmful to approximately neutral. The
single pro-hypothesis result (industrial production, latent-regime
model) is narrow and does not survive a more careful factor.

## The crucial caveat (why this is "absence of evidence")

The test is underpowered exactly where it matters:

- The **rich soft block exists only post-2016**, and that window
  contains essentially **one severe stress episode — COVID** — which is
  an atypical (pandemic) shock, not the confidence/financial crisis the
  hypothesis is usually about. Stress cells hold ~8–13 origins.
- The **broad-crisis sample (long window) has only the thinnest soft
  proxy** (Michigan sentiment alone), because nothing else has deep
  real-time vintages.
- Decomposition repeatedly shows the soft penalty (where it exists) is
  **driven by calm periods, not crises** — i.e. the surveys add noise in
  normal times rather than failing specifically in stress.
- The **marquee business surveys the original claim is built on — ISM/PMI
  and Conference Board confidence — are not available on FRED with
  real-time vintages**, so they are absent from our soft block entirely.

So the defensible statement is: *we cannot show soft data systematically
helps (or hurts) a regime-aware hard-data model around crises*, not *soft
data does not help*.

## Bottom line

> Modeling the volatility regime is what buys crisis-period forecast
> accuracy, and it does so chiefly through hard data. We find no robust
> evidence that this set of soft indicators adds incremental value in
> volatile periods — but with few well-identified stress episodes in the
> rich-soft sample and the leading business surveys unavailable in real
> time, this is best read as absence of evidence rather than a refutation.

## Limitations / what would sharpen the answer

- **More crisis episodes with rich soft data.** Either back-cast the
  soft block with (non-real-time) final data for 2008/2011 as an
  explicit robustness arm, or extend the sample forward over time.
- **The missing surveys.** ISM and Conference Board require paid
  sources; adding them is the most direct way to test the canonical
  version of the claim.
- **Alternative stress definitions.** VIX stress ≈ COVID here; an NFCI-
  or recession-based regime might capture confidence crises where
  surveys are theorized to win (currently a robustness stub).
- **Cross-country replication** (euro area: ESI, area PMI, Eurostat,
  ECB SPF) — designed but not executed.

## Reproducibility

All results regenerate offline from cached inputs + parquet forecasts.
Models in `models/`, evaluation harness and GW/DM tests in
`evaluation/`, run scripts in `scripts/`, narrative in `notebooks/00–06`.
Every forecasting model carries numerical safeguards (stationarity
screens, FFR ≥ 0 screen) and was verified on stress origins before any
RMSE was trusted; percentage-improvement figures are read alongside
absolute RMSE because hard-model denominators are tiny in stress cells.
