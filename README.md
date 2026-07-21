# Solar Thermophotovoltaic (STPV) System Simulation

A numerical Python model of a Solar Thermophotovoltaic (STPV) energy conversion system — from concentrated sunlight, through a thermally-equilibrated absorber/emitter cavity, to a photovoltaic cell.

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
