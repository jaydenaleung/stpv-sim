# Solar Thermophotovoltaic (STPV) System Simulation

A numerical Python model of a Solar Thermophotovoltaic (STPV) energy conversion system — from concentrated sunlight, through a thermally-equilibrated absorber/emitter cavity, to a photovoltaic cell.

This simulation was designed for a research project at the Secondary Student Training Program by the Belin-Blank Center, University of Iowa. You can view the full research poster [here](https://drive.google.com/file/d/1Hyl38rfJYm-bjaMDOagI03IL1U8eMvh3/view?usp=drivesdk).

## What is STPV?

A Solar Thermophotovoltaic system concentrates sunlight onto an absorber, which heats up and re-radiates that energy as thermal (infrared/near-infrared) light. An emitter — often the same physical part as the absorber, or a separate surface radiatively coupled to it — shapes that thermal spectrum and sends it to a photovoltaic cell tuned to harvest it. The appeal is that a well-designed selective absorber/emitter pair can reshape broadband sunlight into a narrower spectral band matched to the PV cell's bandgap, in principle enabling higher conversion efficiencies than direct photovoltaics, and can keep producing power for a while after the sun goes behind a cloud (thermal storage) since the cavity stays hot.

## Purpose and goals

This project is a self-contained numerical model that:

- Solves for the **thermal equilibrium temperature** of the absorber/emitter cavity under concentrated sunlight, using real (or simplified) spectral emissivity data.
- Computes the **optical, spectral, and electrical efficiency chain** from incident sunlight down to delivered electrical power.
- Lets you configure a full physical system — concentrator, absorber/emitter, optional filter, PV cell — with real-world engineering parameters (diameters, focal lengths, material choices, dark current, etc.) rather than abstract knobs.
- Generates the parametric sweep plots needed to reason about design trade-offs (concentration vs. temperature, area ratio, dark current tolerance, bandgap choice).

It is meant as an engineering tool for STPV design choices, not a substitute for a full multi-physics (radiative transfer + conduction + convection) simulation.

## Theory

The governing power-balance equations, integrals, and equilibrium conditions this model implements are laid out in **[THEORY.md](THEORY.md)**. In brief:

- **Incident solar power** `P_sol` — concentrated AM1.5 sunlight absorbed by the absorber, integrated over wavelength and incident angle.
- **Absorber re-radiation** `P_abs(T)` and **emitter output** `P_emit(T)` — thermal (Planck's law) emission from each surface, weighted by their spectral/angular emissivity.
- **Thermal equilibrium**: the temperature `T_eq` at which `P_sol = P_abs(T) + β·P_emit(T)`, where `β` is the emitter-to-absorber area ratio — found numerically via root-finding.
- **Efficiency chain**: overall system efficiency `η_STPV = η_int · η_pvc`, where `η_int` is the intermediate (thermal/spectral) efficiency and `η_pvc = SE · IQE · VF · FF · CE` breaks the PV cell's efficiency into spectral, quantum, voltage, fill-factor, and cavity-coupling factors.

## Features

- **Optical front end**: choice of Fresnel lens or parabolic dish primary concentrator, with an optional compound parabolic concentrator (CPC) second stage.
- **Absorber/emitter modeling**: pick a library material (tungsten, tantalum, graphite, silicon carbide, stainless steel) or supply your own emissivity properties directly; tungsten and silicon carbide additionally support a rigorous transfer-matrix method (TMM) optical model (via the bundled [`tmm`](tmm-master/) package) for real thin-film/dispersive behavior. Absorber and emitter can be the same physical part or two independently sized/configured surfaces.
- **Optional dichroic shortpass filter** between the emitter and the PV cell, modeling spectral photon recycling back to the cavity.
- **Radiative view factors**: emitter → filter → cell geometry (sizes and spacings) is used to compute a real coaxial-disk view factor, which sets the effective cavity coupling efficiency instead of assuming perfect coupling.
- **Silicon PV cell model**: bandgap, IQE, fill factor, dark current (or datasheet Voc/Jsc), cell temperature, and geometry, including a heatsink toggle.
- **Thermal equilibrium solver** with melting-point / runaway-heating safeguards.
- **Five parametric sweep plots**: thermal equilibrium proof, concentration sweep, absorber/emitter area-ratio sweep, dark current sweep, and PV bandgap sweep (blackbody vs. narrowband emitter comparison) — each saved to disk and displayed.
- All user-editable parameters live in one clearly marked block near the top of `stpv_model.py`, so the whole system can be reconfigured without touching the physics code.

## Getting started

```bash
pip install -r requirements.txt
python stpv_model.py
```

This runs a set of internal validation checks (verifying the numerical integrals against known closed-form results), prints the configured system and its computed operating point, and generates/saves the five sweep plots as PNGs in the project directory.

## Validation

The model has been validated against closed-form physics, published analytical results, and two experimental/simulation studies from the STPV literature:

- **[A]** M. A. Abbas et al., "Nanostructured chromium-based broadband absorbers and emitters to realize thermally stable solar thermophotovoltaic systems," *Nanoscale* **14**, 6425 (2022) — the source of the power-balance formulation in `THEORY.md`.
- **[R]** V. D. Rumyantsev et al., "Structural Features of a Solar TPV System," *6th Conf. on Thermophotovoltaic Generation of Electricity* (2004) — an experimental two-stage-concentrator STPV system.

### Closed-form and internal consistency

| Check | Agreement |
|---|---|
| Blackbody `P_emit` vs. Stefan–Boltzmann σT⁴ (500–3400 K) | ≤ 0.11% |
| Planck peak vs. Wien displacement law | ≤ 0.014% |
| AM1.5 total irradiance | 1000 W/m² by construction |
| Max Si photocurrent from AM1.5 | 43.1 mA/cm² (literature 43–46) |
| Angular averaging normalisation (hemispherical and cone) | exact |
| Wavelength-grid convergence vs. a 20× finer grid | 0.00% (smooth), 0.35% (step band edges) |
| Thermodynamic limits (η_int ≤ 1, VF ≤ 1, η_STPV ≤ Carnot) | no violations across N_s = 10–10⁴, β = 0.5–5 |

### Against paper [A]

| Check | Agreement |
|---|---|
| Analytical intermediate efficiency, η_int = 1 − πT⁴/(N_sΩ_sT_s⁴) (their eqn. 9) | ≤ 0.005% at four (T, N_s) points |
| Incidence cone angle θ_c = asin√(N_sΩ_s/π) vs. the model's étendue relation | 0.39% |
| Reported Cr-absorber η_int at 873/1573/1673 K and 500–40 000 suns (their Fig. 2d) | ≤ 3.4% |
| Area-ratio identity η_int = β/(1+β) for a symmetric cavity | exact |
| **End-to-end system efficiency** at 1573 K, 3000 suns, β = 2.25, E_g = 0.54 eV | **model 20.5% vs. paper 21%** |

### Against paper [R]

| Check | Result |
|---|---|
| GaSb photocurrent from a blackbody emitter at 2200 K | 58.6 A/cm² (paper: > 40 A/cm² for equal areas) |
| Emitter-to-absorber area ratio β = 5–10 reduces re-radiation loss | η_int rises 50% → 90.9% as β goes 1 → 10 |
| Tungsten emitter blue-shifts emission vs. a grey body (their Fig. 10) | reproduced; above-bandgap fraction 77.7% vs. 37.2% |
| Photon recycling from the cell's back-surface reflector (return efficiency 90%) | at their β = 10 geometry, η_STPV 25.1% → 27.5% with T_eq rising 2539 → 2582 K |
| Two-stage dish + CPC, 0.45 m² primary at 8000× | reproduced exactly from geometry |

### Material data

Tungsten optical constants use a Lorentz–Drude model with temperature-dependent Drude damping scaled from measured resistivity. Room-temperature reflectance matches handbook values within 1–8% over 0.5–10 µm, and total hemispherical emissivity matches handbook values within 6% from 1000 K to 2800 K. Silicon carbide reproduces the 10.3–12.6 µm reststrahlen band.

### Known limitations

- **Spectral efficiency convention.** `SE` normalises above-bandgap power by the *total* emitted power, so sub-bandgap emission is charged as a loss. Paper [A]'s `U_eff` normalises only by above-bandgap power. The two agree at high temperature (hence the 20.5%/21% match at 1573 K) but diverge sharply when most emission is sub-bandgap — at 873 K the model gives ~5% where [A] reports 15%. Since [A] explicitly ignores photon recycling, the model's convention is the conservative one.
- **Silicon carbide emissivity** uses the opaque `ε = 1 − R` assumption, which is rigorous inside the reststrahlen band but overestimates emittance in SiC's transparent near-infrared window.
- **Dark current is held fixed** across the bandgap sweep, whereas detailed balance gives J₀ ∝ exp(−E_g/kT_c). Testing shows this barely moves the optimum bandgap, but it does inflate efficiency at low E_g.
- Radiative transfer only — no conduction or convection, and the absorber/emitter is treated as isothermal.

## Project structure

| Path | Contents |
|---|---|
| `stpv_model.py` | The complete model: physics, materials, sweeps, and the user input block |
| `THEORY.md` | The mathematical formulation this model implements |
| `INSTRUCTIONS.md` | The original AI build brief/specification for this project |
| `tmm-master/` | Bundled third-party transfer-matrix method library ([sbyrnes321/tmm](https://github.com/sbyrnes321/tmm)) |
| `requirements.txt` | Python dependencies (numpy, scipy, matplotlib) |

## AI disclosure

This project was built collaboratively with an AI coding assistant (Claude Code), working from the specification in **[INSTRUCTIONS.md](INSTRUCTIONS.md)** and the physics in **[THEORY.md](THEORY.md)**. The AI generated and iterated on the implementation — including the numerical integration scheme, material optical models, and plotting — under direction from and review by a human collaborator across a series of feature requests, corrections, and physics sanity checks (e.g. verifying that a compound parabolic concentrator cannot increase delivered concentration onto a fixed-size absorber beyond what energy conservation allows). Values such as material emissivities, optical constants, and default component dimensions are representative/illustrative rather than sourced from a specific real-world datasheet or experiment; treat this as an educational and design-exploration tool, and verify any values you rely on for real engineering decisions.
