# Machine learning in SWRO applications

## Environmental problem
Producing industrial water from seawater relies on SWRO as the main desalination step. SWRO performance (permeate flow, permeate salinity) and energy use (SEC) depend non-linearly on feed flow, temperature, salinity and pressure.

## Dataset source
**Simulated**, not measured: 18,816 scenarios from a simplified solution–diffusion model (`notebooks/swro_sim.py`), one pressure vessel × 7 elements, following the grid of Daaboub et al. (2024), *Enhancing Seawater RO Desalination Efficiency Using Digital Twins and ML*, FAIA 386 (doi:10.3233/FAIA240403). Membrane parameters (A, B, k) are assumed values, not vendor data. Ranges: feed flow 5–17 m³/h, T 10–40 °C, salinity 30–44 g/L, pressure 40–80 bar.

## Research questions
1. Which ML model best predicts permeate flow, permeate salinity and SEC?
2. Which operating variables control each output (SHAP)?
3. What does a single SWRO pass give for the Krabi raw water, and how far is it from ASTM Type II?

## Methods
70/30 split → feature engineering (osmotic pressure, net driving pressure NDP) → 6 models (Linear, Random Forest, XGBoost, LightGBM, CatBoost, ANN/MLP) → Randomized SearchCV (3-fold) → R², RMSE, MAE → SHAP → scenario scan for Krabi water. Salinity and SEC modelled on log scale.

## Main results (test set, R²)
| Model | Permeate flow | Permeate salinity | SEC |
|---|---|---|---|
| Linear | 0.929 | 0.828 | 0.814 |
| Random Forest | 0.9989 | 0.9965 | 0.9956 |
| XGBoost | 0.9997 | 0.9991 | 0.9992 |
| LightGBM | 0.9996 | 0.9985 | 0.9992 |
| **CatBoost** | **0.9999** | **0.9998** | **0.9999** |
| ANN (MLP) | 0.9997 | 0.9978 | 0.9968 |

## Best model
CatBoost for all three targets (consistent with Daaboub et al.). Caveat: the models learn a deterministic simulator, so near-perfect R² mostly shows that the surrogate reproduces the physics, not real-plant accuracy.

## Feature importance (mean |SHAP|)
- Permeate flow: NDP (0.98) ≫ feed flow (0.50) > temperature > salinity
- Permeate salinity (log): temperature (0.24) > feed flow (0.18) > NDP > salinity
- SEC (log): feed flow (0.20) > NDP (0.14) > salinity > temperature

## Environmental interpretation
- Driving force vs osmotic pressure (NDP) controls water production; warm water raises flux but also salt passage.
- water parameter at 39 °C, 35 g/L, 11 m³/h feed, 62–80 bar: permeate TDS ≈ 230–240 mg/L, recovery ≈ 43–58 %, SEC ≈ 4.8–5.0 kWh/m³ (no energy recovery). ASTM Type II needs ≤ 1 µS/cm, so single-pass SWRO is nowhere near enough → 2nd-pass RO + EDI / mixed-bed polishing (and TOC control) are required.
- Raw-water table check: Al = 530 mg/L and K = 3,070 mg/L look abnormally high for seawater (typical ~µg/L and ~400 mg/L) — verify units before using them in the design.

## Limitations
Simulated data; no fouling, scaling, boron, pressure drop, pretreatment or ageing; temperature range capped at 40 °C. Next step: validate with real plant logs.

## Reproduce
`pip install -r requirements.txt` → `python notebooks/swro_sim.py` (from repo root) → run `notebooks/ML_analysis.ipynb`.
