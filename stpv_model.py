#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stpv_model.py -- Numerical model of a Solar Thermophotovoltaic (STPV) system.

Implements the physics described in THEORY.md under the architectural
constraints of INSTRUCTIONS.md:

    sunlight --> concentrator (N_s suns) --> selective absorber --> hot
    isothermal absorber/emitter module (temperature T) --> selective
    emitter --> Si PV cell.

Power bookkeeping (all powers in W per m^2 of absorber aperture):

    P_sol          = 2*pi * int_0^theta_c sin(t)cos(t) dt
                          * int eps_abs(lam, t) I_AM1.5(lam) N_s dlam
    P_abs(T)       = 2*pi * int_0^{pi/2} sin(t)cos(t) dt
                          * int eps_abs(lam, t) I_BB(lam, T) dlam
    P_emit(T)      = 2*pi * int_0^{pi/2} sin(t)cos(t) dt
                          * int eps_emit(lam, t) I_BB(lam, T) dlam
    equilibrium:     P_sol - P_abs(T) - beta * P_emit(T) = 0,
                     beta = A_emit / A_abs

    eta_int  = 1 - P_abs(T_eq) / P_sol
    eta_pvc  = SE * IQE * VF * FF * CE,
               VF = (k_B T_c / (q V_g)) * ln(I_L / I_0 + 1)
    eta_STPV = eta_int * eta_pvc

Modeling choices
----------------
* Spectral, directional emissivities eps(lam, theta) are computed with the
  transfer-matrix method using the bundled `tmm` library by Steven Byrnes
  (https://github.com/sbyrnes321/tmm, in ./tmm-master).  By Kirchhoff's law
  the emissivity of an opaque stack equals its absorptivity, eps = 1 - R,
  averaged over s and p polarizations.
* Tungsten optical constants n + ik come from the Lorentz-Drude fit of
  Rakic et al., Appl. Opt. 37, 5271 (1998); coating dielectrics (HfO2,
  SiO2) are treated as lossless constant-index films.
* The AM1.5 spectrum is a coarse-sampled ASTM G-173 global-tilt shape,
  normalized so that 1 sun integrates to exactly 1 kW/m^2.
* Azimuthal symmetry: the phi integral contributes the factor 2*pi.  With
  the substitution u = sin^2(theta) the polar integral becomes
  2*pi int sin cos dtheta = pi int du, evaluated from precomputed
  eps(lam, u) tables (Gauss-type grids are unnecessary: eps is smooth).
* The incident cone half-angle theta_c is tied to the concentration by
  etendue conservation, sin(theta_c) = sqrt(N_s) * sin(theta_sun), unless
  the user fixes theta_c explicitly.
* Wavelength integrals run over 1e-9 m .. 1e-4 m on a log-spaced grid
  (per INSTRUCTIONS.md, instead of 0..infinity) using
  scipy.integrate.simpson; Planck's law is overflow-guarded.

Running `python stpv_model.py` validates the numerics, solves the default
system, and produces the five required parametric-sweep figures, each saved
to disk as a PNG before being displayed.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Local tmm library (sbyrnes321/tmm).  The folder "tmm-master" is not an
# importable package name, so its directory is put on sys.path and the
# self-contained core module is imported directly.
# ----------------------------------------------------------------------------
_PROJECT_DIR: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_DIR / "tmm-master"))
import tmm_core as tmm  # noqa: E402  (local library, see INSTRUCTIONS.md)

import matplotlib.pyplot as plt  # noqa: E402
from scipy.integrate import cumulative_trapezoid, quad, simpson  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

# ============================================================================
# 1. PHYSICAL CONSTANTS AND GLOBAL NUMERICAL GRIDS
# ============================================================================

H: float = 6.62607015e-34        # Planck constant [J s]
C: float = 2.99792458e8          # speed of light [m/s]
KB: float = 1.380649e-23         # Boltzmann constant [J/K]
Q: float = 1.602176634e-19       # elementary charge [C]
SIGMA: float = 5.670374419e-8    # Stefan-Boltzmann constant [W m^-2 K^-4]

ONE_SUN: float = 1000.0          # 1 sun = 1 kW/m^2 (per INSTRUCTIONS.md)
THETA_SUN: float = np.deg2rad(0.2665)   # half-angle subtended by the sun

T_MIN: float = 250.0             # lower search bound for equilibrium [K]
T_MAX: float = 3500.0            # upper search bound [K] (W melts at 3695 K)
T_MELT_W: float = 3695.0         # tungsten melting point [K]

# Master wavelength grid: 1 nm .. 100 um, log-spaced (INSTRUCTIONS.md 4).
LAM_GRID: np.ndarray = np.geomspace(1.0e-9, 1.0e-4, 2001)


class EquilibriumError(RuntimeError):
    """Raised when no thermal equilibrium exists within the temperature bounds."""


# ============================================================================
# 1b. CONCENTRATOR (OPTICAL FRONT-END) MODELS
# ============================================================================
# The concentrator collects sunlight over a large aperture and focuses it onto
# the much smaller absorber.  Two options are supported: a Fresnel lens or a
# parabolic dish.  For now these parameters are informational (printed to the
# terminal); the estimated optical efficiency and collection area are not yet
# coupled into the P_sol integral -- that wiring is a later step.


@dataclass(frozen=True)
class FresnelLens:
    """Fresnel-lens concentrator geometry and optical efficiency.

    Attributes
    ----------
    focal_length : float
        Distance from the lens to the focal spot [m].
    width, length : float
        Aperture dimensions of the rectangular lens [m].
    efficiency : float
        Estimated optical throughput (0-1): fraction of incident sunlight that
        reaches the absorber after Fresnel reflection and absorption losses.
    """
    focal_length: float
    width: float
    length: float
    efficiency: float
    kind: str = field(default="Fresnel lens", init=False)

    @property
    def collection_area(self) -> float:
        """Light-collecting aperture area A = width * length [m^2]."""
        return self.width * self.length

    def summary_lines(self) -> List[Tuple[str, str]]:
        """(label, formatted value) pairs for terminal printing."""
        return [
            ("focal length", f"{self.focal_length:.3f} m"),
            ("lens width", f"{self.width:.3f} m"),
            ("lens length", f"{self.length:.3f} m"),
            ("aperture area", f"{self.collection_area:.4f} m^2"),
            ("optical efficiency", f"{self.efficiency:.2f} "
                                   f"({100 * self.efficiency:.0f} %)"),
        ]


@dataclass(frozen=True)
class ParabolicDish:
    """Parabolic-dish concentrator geometry and optical efficiency.

    Attributes
    ----------
    focal_length : float
        Focal length of the paraboloid [m].
    diameter : float
        Rim diameter of the circular aperture [m].
    depth : float
        Depth (sagitta) of the dish from rim plane to vertex [m].
    efficiency : float
        Estimated optical throughput (0-1): fraction of incident sunlight that
        reaches the absorber after mirror-reflectance and spillage losses.
    """
    focal_length: float
    diameter: float
    depth: float
    efficiency: float
    kind: str = field(default="Parabolic dish", init=False)

    @property
    def collection_area(self) -> float:
        """Light-collecting aperture area A = pi (D/2)^2 [m^2]."""
        return np.pi * (self.diameter / 2.0) ** 2

    def summary_lines(self) -> List[Tuple[str, str]]:
        """(label, formatted value) pairs for terminal printing."""
        return [
            ("focal length", f"{self.focal_length:.3f} m"),
            ("diameter", f"{self.diameter:.3f} m"),
            ("depth of dish", f"{self.depth:.3f} m"),
            ("aperture area", f"{self.collection_area:.4f} m^2"),
            ("optical efficiency", f"{self.efficiency:.2f} "
                                   f"({100 * self.efficiency:.0f} %)"),
        ]


@dataclass(frozen=True)
class Concentrator:
    """Optical front-end selector: a Fresnel lens OR a parabolic dish.

    `use_fresnel` is the y/n toggle: True selects the Fresnel lens, False the
    parabolic dish.  Both option structs are stored so the choice can be
    flipped without editing geometry.
    """
    use_fresnel: bool
    fresnel: FresnelLens
    dish: ParabolicDish

    @property
    def active(self) -> "FresnelLens | ParabolicDish":
        """The currently selected concentrator option."""
        return self.fresnel if self.use_fresnel else self.dish

    def print_summary(self) -> None:
        """Print the selected concentrator's input parameters to the terminal."""
        active = self.active
        print(f"  concentrator : {active.kind} "
              f"(use_fresnel = {self.use_fresnel})")
        for label, value in active.summary_lines():
            print(f"      {label:<20s}: {value}")


# ============================================================================
# 1c. ABSORBER / EMITTER MATERIAL AND GEOMETRY
# ============================================================================
# The absorber (hot face toward the sun) and emitter (hot face toward the PV)
# may be the same physical part.  Their radiative behaviour is captured either
# by a named material (properties looked up in MATERIAL_LIBRARY) or by
# directly-specified emissivity properties (MaterialProperties).  Tungsten can
# additionally use the detailed transfer-matrix (TMM) model with thin-film AR
# coatings.


@dataclass(frozen=True)
class MaterialProperties:
    """The radiative properties needed to model an absorber/emitter surface.

    A compact two-band emissivity description sufficient for the power
    integrals: a (near-visible) short-wavelength emissivity, a thermal-IR
    long-wavelength emissivity, and the band-edge wavelength separating them.
    Selective emitters (metals) have emissivity_short > emissivity_long; near-
    gray materials (graphite, SiC) set the two roughly equal.

    Attributes
    ----------
    name : str
        Material name (for logs/plots).
    emissivity_short : float
        Emissivity for lambda <= cutoff_wavelength (solar / visible-NIR).
    emissivity_long : float
        Emissivity for lambda > cutoff_wavelength (thermal infrared).
    cutoff_wavelength : float
        Band-edge wavelength separating the two regimes [m].
    """
    name: str
    emissivity_short: float
    emissivity_long: float
    cutoff_wavelength: float


#: Representative high-temperature emissivity properties for common absorber/
#: emitter materials (literature order-of-magnitude values).  Refractory metals
#: (W, Ta) are spectrally selective; graphite and SiC are near-gray; stainless
#: steel is a moderate-emissivity oxidized metal.
MATERIAL_LIBRARY: Dict[str, MaterialProperties] = {
    "tungsten":        MaterialProperties("tungsten",        0.45, 0.10, 2.0e-6),
    "tantalum":        MaterialProperties("tantalum",        0.40, 0.12, 2.0e-6),
    "graphite":        MaterialProperties("graphite",        0.85, 0.85, 2.0e-6),
    "silicon carbide": MaterialProperties("silicon carbide", 0.90, 0.88, 2.0e-6),
    "stainless steel": MaterialProperties("stainless steel", 0.40, 0.30, 2.0e-6),
}


@dataclass(frozen=True)
class AbsorberEmitter:
    """Absorber/emitter material selection and rectangular-prism geometry.

    Attributes
    ----------
    same_part : bool
        If True, the absorber and emitter are the same piece/material, so the
        emitter surface uses the absorber's emissivity model and (with equal
        face areas) the nominal area ratio beta = 1.
    material : str
        Material name; used as a key into MATERIAL_LIBRARY (case-insensitive).
    use_tmm : bool
        If True and material == "tungsten" (and no custom_properties), use the
        detailed TMM thin-film model instead of the two-band approximation.
    custom_properties : MaterialProperties or None
        Directly-specified radiative properties.  If provided, these override
        the material lookup entirely (the "input the properties" path).
    length, width, depth : float
        Rectangular-prism dimensions [m].  length x width is the absorber face
        area (1-5 cm^2 per THEORY.md); depth is the thickness (0.1-1 cm).
    emitter_length, emitter_width : float or None
        Optional distinct emitter-face dimensions [m] when same_part is False;
        default to the absorber face when None.
    """
    same_part: bool = False
    material: str = "tungsten"
    use_tmm: bool = True
    custom_properties: Optional[MaterialProperties] = None
    length: float = 0.02
    width: float = 0.02
    depth: float = 0.005
    emitter_length: Optional[float] = None
    emitter_width: Optional[float] = None

    @property
    def absorber_area(self) -> float:
        """Absorber face area A_abs = length * width [m^2]."""
        return self.length * self.width

    @property
    def emitter_area(self) -> float:
        """Emitter face area A_emit [m^2] (equals A_abs unless overridden)."""
        el = self.length if self.emitter_length is None else self.emitter_length
        ew = self.width if self.emitter_width is None else self.emitter_width
        return el * ew

    @property
    def nominal_beta(self) -> float:
        """Geometry-derived area ratio beta = A_emit / A_abs."""
        return self.emitter_area / self.absorber_area

    def resolved_properties(self) -> Optional[MaterialProperties]:
        """Return the effective MaterialProperties (custom or library lookup)."""
        if self.custom_properties is not None:
            return self.custom_properties
        return MATERIAL_LIBRARY.get(self.material.lower())

    def build(self, verbose: bool = True) -> Tuple["Emissivity", "Emissivity"]:
        """Construct (absorber, emitter) emissivity models from this config.

        Tungsten with use_tmm uses the detailed TMM stacks (bare W on both
        faces when same_part, else a solar-optimized AR absorber and an
        NIR-enhanced emitter).  All other cases use the two-band analytic model
        built from the resolved MaterialProperties.
        """
        mat = self.material.lower()
        if self.use_tmm and mat == "tungsten" and self.custom_properties is None:
            if self.same_part:
                shared = TMMEmissivity(LayerStack(
                    "bare tungsten (absorber = emitter)", (), "W"),
                    verbose=verbose)
                return shared, shared
            absorber = TMMEmissivity(LayerStack(
                "W + 80 nm HfO2 AR (selective absorber)",
                coatings=(("HfO2", 80e-9),), substrate="W"), verbose=verbose)
            emitter = TMMEmissivity(LayerStack(
                "W + 140 nm HfO2 (NIR-enhanced emitter)",
                coatings=(("HfO2", 140e-9),), substrate="W"), verbose=verbose)
            return absorber, emitter

        props = self.resolved_properties()
        if props is None:
            raise ValueError(
                f"Unknown absorber/emitter material '{self.material}'. "
                f"Choose one of {sorted(MATERIAL_LIBRARY)} or supply "
                f"custom_properties.")
        absorber = two_band_emissivity(props, f"{props.name} absorber")
        emitter = (absorber if self.same_part
                   else two_band_emissivity(props, f"{props.name} emitter"))
        return absorber, emitter

    def print_summary(self) -> None:
        """Print the absorber/emitter material and geometry to the terminal."""
        props = self.resolved_properties()
        src = ("custom properties" if self.custom_properties is not None
               else f"library material '{self.material}'")
        model = ("TMM thin-film" if (self.use_tmm and self.material.lower()
                 == "tungsten" and self.custom_properties is None)
                 else "two-band analytic")
        print(f"  absorber/emitter : {src}, {model} model")
        print(f"      same part            : {self.same_part}")
        if props is not None and model != "TMM thin-film":
            print(f"      emissivity (short)   : {props.emissivity_short:.2f} "
                  f"(lambda <= {props.cutoff_wavelength * 1e6:.2g} um)")
            print(f"      emissivity (long)    : {props.emissivity_long:.2f} "
                  f"(lambda >  {props.cutoff_wavelength * 1e6:.2g} um)")
        print(f"      length x width       : {self.length * 100:.2f} x "
              f"{self.width * 100:.2f} cm  (A_abs = "
              f"{self.absorber_area * 1e4:.2f} cm^2)")
        print(f"      depth (thickness)    : {self.depth * 100:.2f} cm")
        print(f"      nominal beta         : {self.nominal_beta:.3f} "
              f"(A_emit / A_abs)")


# ============================================================================
# 1d. OPTIONAL DICHROIC SHORTPASS FILTER (cold-side photon recycling)
# ============================================================================


@dataclass(frozen=True)
class DichroicFilter:
    """Optional shortpass filter between the emitter and the PV cell.

    A shortpass dichroic transmits above-bandgap (short-wavelength) photons to
    the cell and reflects sub-bandgap (long-wavelength) photons back to the
    emitter, recycling that energy.  Modeled by a spectral transmittance
    tau(lambda): the reflected fraction 1 - tau returns to the emitter and so
    does not count as a net radiative loss, which raises T_eq and the spectral
    efficiency.

    Attributes
    ----------
    enabled : bool
        y/n: whether the filter is present.
    cutoff_wavelength : float
        Shortpass edge [m]; lambda <= cutoff is transmitted (passband).
    passband_transmittance : float
        Transmittance tau for lambda <= cutoff (useful above-gap light).
    stopband_transmittance : float
        Transmittance tau for lambda > cutoff; the remainder is reflected back
        to the emitter (photon recycling).
    spacing : float
        Emitter-to-filter gap [m] (informational; would set a view factor).
    diameter, thickness : float
        Filter aperture diameter and substrate thickness [m] (informational).
    """
    enabled: bool = False
    cutoff_wavelength: float = 1.2e-6
    passband_transmittance: float = 0.95
    stopband_transmittance: float = 0.05
    spacing: float = 0.005
    diameter: float = 0.03
    thickness: float = 0.002

    def transmittance(self, lam: np.ndarray) -> np.ndarray:
        """Spectral transmittance tau(lambda); all-ones when disabled."""
        lam = np.asarray(lam, dtype=float)
        if not self.enabled:
            return np.ones_like(lam)
        return np.where(lam <= self.cutoff_wavelength,
                        self.passband_transmittance,
                        self.stopband_transmittance)

    def print_summary(self) -> None:
        """Print the filter configuration to the terminal."""
        if not self.enabled:
            print("  dichroic filter  : none (not used)")
            return
        print("  dichroic filter  : shortpass (enabled)")
        print(f"      cutoff wavelength    : {self.cutoff_wavelength * 1e6:.3f} um")
        print(f"      passband tau         : {self.passband_transmittance:.2f} "
              f"(lambda <= cutoff, to cell)")
        print(f"      stopband tau         : {self.stopband_transmittance:.2f} "
              f"(lambda >  cutoff; rest recycled)")
        print(f"      emitter-filter gap   : {self.spacing * 1e3:.2f} mm")
        print(f"      diameter x thickness : {self.diameter * 1e3:.1f} x "
              f"{self.thickness * 1e3:.2f} mm")


# ============================================================================
# 1e. PV CELL
# ============================================================================


@dataclass(frozen=True)
class PVCell:
    """Single-junction PV cell parameters (default: silicon, 1.12 eV).

    Attributes
    ----------
    bandgap_ev : float
        Bandgap energy E_g [eV]; also the bandgap voltage V_g [V].
    iqe : float
        Internal quantum efficiency (dimensionless).
    fill_factor : float
        Fill factor FF from the manufacturer (dimensionless).
    cavity_eff : float
        Cavity efficiency CE (fraction of emitted photons reaching the cell).
    cell_temp : float
        Cell temperature T_c [K].
    dark_current : float
        Dark (saturation) current density J_0 [A/m^2].  Photocurrent and dark
        current enter VF only through the ratio I_L / I_0, so consistent
        current *densities* are used throughout (1 A/m^2 = 1e-4 A/cm^2).
    length, width, depth : float
        Cell dimensions [m]; length x width is the active face area (1-5 cm^2).
    filter_spacing : float
        Gap between the filter (or emitter) and the cell [m] (informational).
    rated_voc, rated_jsc : float or None
        Datasheet open-circuit voltage [V] and short-circuit current density
        [A/m^2].  If both are given, the effective dark current is derived from
        them via the diode law J_0 = J_sc / (exp(q V_oc / k T_c) - 1), so V_oc
        and J_sc drive VF directly; otherwise `dark_current` is used.
    rated_efficiency : float or None
        Datasheet cell efficiency [-] (informational/reference only).
    heatsink : bool
        Whether a heatsink holds the cell at `cell_temp` (thermal management).
    """
    bandgap_ev: float = 1.12
    iqe: float = 0.95
    fill_factor: float = 0.80
    cavity_eff: float = 0.95
    cell_temp: float = 300.0
    dark_current: float = 1.0e-8   # = 1e-12 A/cm^2, a good silicon cell
    length: float = 0.02
    width: float = 0.02
    depth: float = 0.0003          # 300 um wafer
    filter_spacing: float = 0.005
    rated_voc: Optional[float] = None
    rated_jsc: Optional[float] = None
    rated_efficiency: Optional[float] = None
    heatsink: bool = True

    @property
    def bandgap_j(self) -> float:
        """Bandgap energy [J]."""
        return self.bandgap_ev * Q

    @property
    def bandgap_wavelength(self) -> float:
        """Bandgap wavelength lam_g = h c / E_g [m]."""
        return H * C / self.bandgap_j

    @property
    def active_area(self) -> float:
        """Active cell face area A_cell = length * width [m^2]."""
        return self.length * self.width

    @property
    def effective_dark_current(self) -> float:
        """Dark current density J_0 driving VF [A/m^2].

        Derived from the rated V_oc and J_sc via the diode law when both are
        provided; otherwise the explicit `dark_current` field is used.
        """
        if self.rated_voc is not None and self.rated_jsc is not None:
            denom = np.expm1(Q * self.rated_voc / (KB * self.cell_temp))
            if denom > 0.0:
                return float(self.rated_jsc / denom)
        return self.dark_current

    def print_summary(self) -> None:
        """Print the PV-cell configuration to the terminal."""
        print(f"  PV cell (Si)     : E_g = {self.bandgap_ev:g} eV, "
              f"T_c = {self.cell_temp:.0f} K"
              f"{' (heatsink)' if self.heatsink else ''}")
        print(f"      active L x W x D     : {self.length * 100:.2f} x "
              f"{self.width * 100:.2f} x {self.depth * 1e3:.2f} mm-D "
              f"(A_cell = {self.active_area * 1e4:.2f} cm^2)")
        print(f"      filter-cell spacing  : {self.filter_spacing * 1e3:.2f} mm")
        print(f"      IQE / FF / CE        : {self.iqe:.2f} / "
              f"{self.fill_factor:.2f} / {self.cavity_eff:.2f}")
        if self.rated_voc is not None and self.rated_jsc is not None:
            print(f"      rated Voc / Jsc      : {self.rated_voc:.3f} V / "
                  f"{self.rated_jsc:.1f} A/m^2  -> J_0 = "
                  f"{self.effective_dark_current:.2e} A/m^2")
        else:
            print(f"      dark current J_0     : "
                  f"{self.effective_dark_current:.2e} A/m^2")
        if self.rated_efficiency is not None:
            print(f"      rated efficiency     : "
                  f"{100 * self.rated_efficiency:.1f} %")


# ============================================================================
# 1f. RADIATIVE VIEW FACTORS (emitter -> filter -> cell geometry)
# ============================================================================
# Finite parallel surfaces separated by a gap intercept only a fraction of each
# other's radiation; the rest spills to the surroundings.  That fraction is the
# view factor, and it sets the geometric part of the cavity efficiency CE.


def view_factor_coaxial_disks(r1: float, r2: float, gap: float) -> float:
    """View factor F_(1->2) between two coaxial parallel disks.

    Parameters
    ----------
    r1, r2 : float
        Disk radii [m] (source disk 1, target disk 2).
    gap : float
        Axial separation between the disks [m].

    Returns
    -------
    float
        Fraction of diffuse radiation leaving disk 1 that reaches disk 2, from
        the exact analytic result (Incropera & DeWitt, Table 13.2).  Approaches
        1 as gap -> 0 and 0 as gap -> infinity.
    """
    if gap <= 0.0:
        return 1.0
    r1_n = r1 / gap
    r2_n = r2 / gap
    s = 1.0 + (1.0 + r2_n * r2_n) / (r1_n * r1_n)
    val = 0.5 * (s - np.sqrt(max(s * s - 4.0 * (r2_n / r1_n) ** 2, 0.0)))
    return float(np.clip(val, 0.0, 1.0))


def _equal_area_radius(area: float) -> float:
    """Radius [m] of the disk having the given area [m^2] (r = sqrt(A/pi))."""
    return float(np.sqrt(area / np.pi))


def system_view_factor(absorber_emitter: "AbsorberEmitter",
                       dichroic_filter: "DichroicFilter",
                       pv_cell: "PVCell") -> float:
    """Geometric fraction of emitter radiation reaching the PV cell.

    The emitter and cell faces are approximated as equal-area coaxial disks and
    the filter as its physical disk aperture.  With a filter present the
    transfer is the product of the two interception steps (emitter->filter and
    filter->cell over their respective spacings); without a filter it is the
    single emitter->cell step over the cell standoff `filter_spacing`.
    """
    r_e = _equal_area_radius(absorber_emitter.emitter_area)
    r_c = _equal_area_radius(pv_cell.active_area)
    if dichroic_filter.enabled:
        r_f = 0.5 * dichroic_filter.diameter
        f_ef = view_factor_coaxial_disks(r_e, r_f, dichroic_filter.spacing)
        f_fc = view_factor_coaxial_disks(r_f, r_c, pv_cell.filter_spacing)
        return f_ef * f_fc
    return view_factor_coaxial_disks(r_e, r_c, pv_cell.filter_spacing)


# ============================================================================
# USER INPUT PARAMETERS  <-- edit this block to configure the whole system
# ============================================================================
# All four subsystems are configured here.  Geometry is in metres; efficiencies
# and emissivities are dimensionless fractions (0-1).  These names are read at
# call time by build_default_system(), so this block may live anywhere below the
# dataclass definitions it uses (Concentrator, AbsorberEmitter, DichroicFilter,
# PVCell) -- it is placed here, near the top, for easy editing.

# --- Solar concentration N_s --------------------------------------------------
# By default N_s is DERIVED from the concentrator aperture / absorber-face area
# ratio (the geometric concentration, computed at the end of this block once the
# concentrator and absorber sizes below are known).  To pin a specific value
# instead, set NS_OVERRIDE to a number of suns; leave it None to use geometry.
NS_OVERRIDE: Optional[float] = None

# --- 1. Concentrator: pick a Fresnel lens or a parabolic dish -----------------
CONCENTRATOR: Concentrator = Concentrator(
    use_fresnel=False,                    # y/n toggle: True -> Fresnel lens
    fresnel=FresnelLens(
        focal_length=0.30, width=0.30, length=0.30, efficiency=0.85),
    dish=ParabolicDish(
        focal_length=0.60, diameter=1.50, depth=0.234, efficiency=0.25),
)

# --- 2. Absorber / emitter: material + geometry -------------------------------
# material: one of MATERIAL_LIBRARY ("tungsten", "tantalum", "graphite",
# "silicon carbide", "stainless steel"), or set custom_properties to input the
# emissivity properties directly.  use_tmm=True gives tungsten the detailed
# thin-film model; same_part=True makes the emitter identical to the absorber.
ABSORBER_EMITTER: AbsorberEmitter = AbsorberEmitter(
    same_part=True,
    material="silicon carbide",
    use_tmm=False,
    custom_properties=None,              # e.g. MaterialProperties("myMat", 0.9, 0.2, 2e-6)
    length=0.02, width=0.02, depth=0.003,
)

# --- 3. Optional dichroic shortpass filter (cold-side photon recycling) -------
DICHROIC_FILTER: DichroicFilter = DichroicFilter(
    enabled=False,                       # y/n toggle
    cutoff_wavelength=1.10e-6,           # ~Si bandgap edge
    passband_transmittance=0.95,
    stopband_transmittance=0.05,
    spacing=0.005, diameter=0.03, thickness=0.002,
)

# --- 4. PV cell (silicon) + heatsink ------------------------------------------
# Provide rated_voc and rated_jsc together to drive VF from datasheet specs,
# or leave them None to use dark_current directly.
PV_CELL: PVCell = PVCell(
    bandgap_ev=1.12, iqe=0.95, fill_factor=0.80, cavity_eff=0.95,
    cell_temp=300.0, dark_current=1.0e-8,
    length=0.02, width=0.02, depth=0.0003,
    filter_spacing=0.005,
    rated_voc=None, rated_jsc=None, rated_efficiency=0.22,
    heatsink=True,
)

# --- Cavity efficiency from geometry ------------------------------------------
# When True, the effective cavity efficiency is the cell's intrinsic cavity_eff
# times the geometric view factor computed from the emitter/filter/cell sizes
# and the spacings above (DichroicFilter.spacing and PVCell.filter_spacing).
# Larger gaps -> more spillage -> lower CE.  Set False to use cavity_eff alone.
USE_VIEW_FACTOR_CE: bool = True

# --- Derived N_s --------------------------------------------------------------
# Geometric concentration = concentrator aperture area / absorber face area.
# This is the raw optical concentration set purely by the two sizes; the
# concentrator's optical efficiency is applied separately (it scales P_sol).
GEOMETRIC_CONCENTRATION: float = (
    CONCENTRATOR.active.collection_area / ABSORBER_EMITTER.absorber_area)
NS_DEFAULT: float = (float(NS_OVERRIDE) if NS_OVERRIDE is not None
                     else GEOMETRIC_CONCENTRATION)
SYSTEM_VIEW_FACTOR: float = (
    system_view_factor(ABSORBER_EMITTER, DICHROIC_FILTER, PV_CELL)
    if USE_VIEW_FACTOR_CE else 1.0)


# ============================================================================
# 2. SPECTRA: PLANCK'S LAW AND THE AM1.5 SOLAR SPECTRUM
# ============================================================================

def planck_spectral_radiance(lam: np.ndarray, temperature: float) -> np.ndarray:
    """Blackbody spectral radiance I_BB(lam, T).

    Parameters
    ----------
    lam : np.ndarray
        Vacuum wavelength [m].
    temperature : float
        Absolute temperature [K].

    Returns
    -------
    np.ndarray
        Spectral radiance [W m^-2 m^-1 sr^-1].  The hemispherical spectral
        emissive power of a blackbody is pi * I_BB, and
        pi * int I_BB dlam = sigma T^4.

    Notes
    -----
    Overflow-guarded: where h c / (lam k_B T) > 700 the value underflows to
    exactly 0 instead of overflowing `exp`; `expm1` avoids catastrophic
    cancellation in the Rayleigh-Jeans limit.
    """
    lam = np.asarray(lam, dtype=float)
    x = H * C / (lam * KB * float(temperature))
    radiance = np.zeros_like(lam)
    ok = x < 700.0
    radiance[ok] = (2.0 * H * C**2 / lam[ok] ** 5) / np.expm1(x[ok])
    return radiance


# Coarse-sampled ASTM G-173 (AM1.5 global tilt) spectral shape.
# Columns: wavelength [um], spectral irradiance [W m^-2 nm^-1] (approximate;
# the absolute scale is irrelevant because the spectrum is renormalized so
# that its integral is exactly ONE_SUN = 1000 W/m^2).
_AM15_LAM_UM: np.ndarray = np.array([
    0.280, 0.300, 0.320, 0.340, 0.360, 0.380, 0.400, 0.450, 0.500, 0.550,
    0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.940, 0.980, 1.000,
    1.050, 1.100, 1.130, 1.160, 1.200, 1.250, 1.300, 1.350, 1.400, 1.450,
    1.500, 1.550, 1.600, 1.650, 1.700, 1.750, 1.800, 1.850, 1.900, 1.950,
    2.000, 2.100, 2.200, 2.300, 2.400, 2.500, 2.600, 2.800, 3.000, 3.200,
    3.500, 4.000,
])
_AM15_IRR: np.ndarray = np.array([
    0.000, 0.050, 0.250, 0.360, 0.480, 0.600, 1.100, 1.600, 1.550, 1.530,
    1.470, 1.430, 1.350, 1.230, 1.110, 0.970, 0.880, 0.550, 0.750, 0.730,
    0.670, 0.600, 0.300, 0.470, 0.480, 0.460, 0.410, 0.150, 0.020, 0.150,
    0.300, 0.280, 0.240, 0.220, 0.200, 0.150, 0.050, 0.010, 0.005, 0.020,
    0.070, 0.080, 0.070, 0.050, 0.040, 0.020, 0.001, 0.001, 0.015, 0.010,
    0.008, 0.003,
])


def _am15_unnormalized(lam: np.ndarray) -> np.ndarray:
    """AM1.5 spectral irradiance before renormalization [W m^-2 m^-1]."""
    return np.interp(np.asarray(lam) * 1e6, _AM15_LAM_UM, _AM15_IRR,
                     left=0.0, right=0.0) * 1e9  # per-nm -> per-m


# Normalize once on a dense linear grid so that 1 sun == 1000 W/m^2 exactly.
_dense = np.linspace(0.25e-6, 4.2e-6, 40001)
_AM15_SCALE: float = ONE_SUN / float(np.trapezoid(_am15_unnormalized(_dense), _dense))
del _dense


def am15_spectral_irradiance(lam: np.ndarray) -> np.ndarray:
    """AM1.5 spectral irradiance at 1 sun, normalized to 1 kW/m^2 total.

    Parameters
    ----------
    lam : np.ndarray
        Vacuum wavelength [m].

    Returns
    -------
    np.ndarray
        Spectral irradiance [W m^-2 m^-1]; zero outside 0.28-4.0 um.
    """
    return _AM15_SCALE * _am15_unnormalized(lam)


# ============================================================================
# 3. MATERIAL OPTICAL CONSTANTS (for TMM)
# ============================================================================

# Lorentz-Drude parameters for tungsten from Rakic et al., Appl. Opt. 37,
# 5271 (1998).  Energies in eV: plasma energy wp, Drude term (f0, Gamma0),
# and four Lorentz oscillators (f_j, Gamma_j, omega_j).
_W_WP: float = 13.22
_W_F0: float = 0.206
_W_G0: float = 0.064
_W_OSC: Tuple[Tuple[float, float, float], ...] = (
    (0.054, 0.530, 1.004),
    (0.166, 1.281, 1.917),
    (0.706, 3.332, 3.580),
    (2.590, 5.836, 7.498),
)


def refractive_index_tungsten(lam: np.ndarray) -> np.ndarray:
    """Complex refractive index n + ik of tungsten (Lorentz-Drude model).

    Parameters
    ----------
    lam : np.ndarray
        Vacuum wavelength [m].

    Returns
    -------
    np.ndarray
        Complex refractive index (Im >= 0, passive medium).
    """
    w = H * C / (Q * np.asarray(lam, dtype=float))  # photon energy [eV]
    eps = np.full_like(w, 1.0, dtype=complex)
    eps -= _W_F0 * _W_WP**2 / (w**2 + 1j * _W_G0 * w)          # Drude
    for f_j, g_j, w_j in _W_OSC:                                # Lorentz
        eps += f_j * _W_WP**2 / ((w_j**2 - w**2) - 1j * g_j * w)
    # Principal sqrt keeps Im(n) >= 0 because Im(eps) > 0 for a passive metal.
    return np.sqrt(eps)


def _constant_index(n: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return a dispersion-free refractive-index function n(lam) = n + 0i."""
    def index(lam: np.ndarray) -> np.ndarray:
        return np.full(np.shape(lam), complex(n))
    return index


#: Registry of material dispersion functions, lam [m] -> complex index.
REFRACTIVE_INDEX: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Air": _constant_index(1.00),
    "W": refractive_index_tungsten,
    "HfO2": _constant_index(1.90),
    "SiO2": _constant_index(1.45),
}


@dataclass(frozen=True)
class LayerStack:
    """An opaque thin-film stack: air / coatings / semi-infinite substrate.

    Attributes
    ----------
    name : str
        Human-readable description used in logs and plot legends.
    coatings : tuple of (material, thickness_m)
        Thin films from front (illuminated side) to back.
    substrate : str
        Optically thick substrate material (default tungsten).
    """
    name: str
    coatings: Tuple[Tuple[str, float], ...] = ()
    substrate: str = "W"


# ============================================================================
# 4. EMISSIVITY MODELS (TMM-based and analytic)
# ============================================================================

class Emissivity:
    """Interface: angle-averaged spectral emissivity/absorptivity models."""

    name: str = "emissivity"

    def hemispherical(self, lam: np.ndarray) -> np.ndarray:
        """Hemispherically averaged spectral emissivity eps_bar(lam).

        Defined as 2 int_0^{pi/2} eps(lam, t) sin(t)cos(t) dt, i.e. the
        weighting that makes  P = pi * int eps_bar I_BB dlam  exact.
        """
        raise NotImplementedError

    def cone_average(self, lam: np.ndarray, theta_c: float) -> np.ndarray:
        """Spectral emissivity averaged over a cone of half-angle theta_c.

        Defined as [2 int_0^{theta_c} eps sin cos dt] / sin^2(theta_c), so a
        perfect absorber returns exactly 1 for any cone.
        """
        raise NotImplementedError


class AnalyticEmissivity(Emissivity):
    """Angle-independent emissivity given by an arbitrary function of lam."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str) -> None:
        self._fn = fn
        self.name = name

    def hemispherical(self, lam: np.ndarray) -> np.ndarray:
        """See base class; angle-independent, so simply eps(lam)."""
        return np.asarray(self._fn(np.asarray(lam)), dtype=float)

    def cone_average(self, lam: np.ndarray, theta_c: float) -> np.ndarray:
        """See base class; angle-independent, so simply eps(lam)."""
        return self.hemispherical(lam)


def gray_emissivity(eps: float, name: Optional[str] = None) -> AnalyticEmissivity:
    """Gray (wavelength/angle independent) emissivity; eps=1 is a blackbody."""
    return AnalyticEmissivity(lambda lam: np.full(np.shape(lam), float(eps)),
                              name or f"gray body (eps = {eps:g})")


def step_band_emissivity(eps_in: float, eps_out: float,
                         lam_lo: float, lam_hi: float,
                         name: Optional[str] = None) -> AnalyticEmissivity:
    """Step-function band emitter: eps_in inside [lam_lo, lam_hi], else eps_out.

    Reproduces the THEORY.md numerical-test emitter, e.g.
    step_band_emissivity(0.85, 0.05, 1.5e-6, 2.4e-6).
    """
    def fn(lam: np.ndarray) -> np.ndarray:
        lam = np.asarray(lam)
        return np.where((lam >= lam_lo) & (lam <= lam_hi), eps_in, eps_out)
    label = name or (f"step emitter (eps = {eps_in:g} in "
                     f"{lam_lo * 1e6:.2g}-{lam_hi * 1e6:.2g} um, else {eps_out:g})")
    return AnalyticEmissivity(fn, label)


def two_band_emissivity(props: "MaterialProperties",
                        name: Optional[str] = None) -> AnalyticEmissivity:
    """Two-band selective emissivity from MaterialProperties.

    emissivity_short applies for lambda <= cutoff_wavelength and
    emissivity_long for longer wavelengths -- i.e. a single-edge step emitter.
    """
    return step_band_emissivity(
        props.emissivity_short, props.emissivity_long,
        0.0, props.cutoff_wavelength,
        name or (f"{props.name} (eps {props.emissivity_short:g}/"
                 f"{props.emissivity_long:g} @ "
                 f"{props.cutoff_wavelength * 1e6:.2g} um)"))


class TMMEmissivity(Emissivity):
    """Spectral-directional emissivity of an opaque stack via the TMM.

    On construction, eps(lam, theta) = 1 - [R_s + R_p]/2 (Kirchhoff's law for
    an opaque multilayer) is evaluated with `tmm.coh_tmm` on a log-spaced
    wavelength grid and a uniform grid in u = sin^2(theta).  Angular
    integrals with the radiative weight sin(t)cos(t) dt = du/2 then reduce to
    cumulative-trapezoid lookups, so all downstream power integrals are
    vectorized and the expensive TMM sweep runs exactly once per stack.
    """

    def __init__(self, stack: LayerStack,
                 lam_grid: Optional[np.ndarray] = None,
                 n_angles: int = 33,
                 verbose: bool = True) -> None:
        self.stack = stack
        self.name = stack.name
        self._lam = (np.geomspace(2.0e-7, 5.0e-5, 192)
                     if lam_grid is None else np.asarray(lam_grid, dtype=float))
        self._u = np.linspace(0.0, 1.0, n_angles)
        # Avoid exactly-grazing incidence (numerically singular Fresnel case);
        # the u = 1 endpoint carries zero radiative weight anyway.
        thetas = np.arcsin(np.sqrt(np.minimum(self._u, 0.999999)))

        d_list: List[float] = [np.inf] + [d for _, d in stack.coatings] + [np.inf]
        materials = ["Air"] + [m for m, _ in stack.coatings] + [stack.substrate]
        n_table = np.column_stack(
            [REFRACTIVE_INDEX[m](self._lam) for m in materials])

        if verbose:
            print(f"  [TMM] computing eps(lam, theta) for '{stack.name}' "
                  f"({self._lam.size} wavelengths x {n_angles} angles x 2 pol)...")
        t0 = time.perf_counter()
        eps = np.empty((self._lam.size, self._u.size))
        for i, lam_i in enumerate(self._lam):
            n_list = list(n_table[i])
            for j, th in enumerate(thetas):
                r_s = tmm.coh_tmm("s", n_list, d_list, th, lam_i)["R"]
                r_p = tmm.coh_tmm("p", n_list, d_list, th, lam_i)["R"]
                eps[i, j] = 1.0 - 0.5 * (r_s + r_p)
        if verbose:
            print(f"  [TMM] done in {time.perf_counter() - t0:.1f} s")

        # Cumulative integral over u, per wavelength: C(lam, u) = int_0^u eps du'
        self._eps = eps
        self._cum = cumulative_trapezoid(eps, self._u, axis=1, initial=0.0)
        self._hemi = self._cum[:, -1]           # int_0^1 eps du

    def hemispherical(self, lam: np.ndarray) -> np.ndarray:
        """See base class (values clamped outside the internal TMM grid)."""
        return np.interp(np.asarray(lam), self._lam, self._hemi)

    def cone_average(self, lam: np.ndarray, theta_c: float) -> np.ndarray:
        """See base class (values clamped outside the internal TMM grid)."""
        u_c = float(np.clip(np.sin(theta_c) ** 2, 1e-12, 1.0))
        cum_at_uc = np.array(
            [np.interp(u_c, self._u, row) for row in self._cum])
        return np.interp(np.asarray(lam), self._lam, cum_at_uc / u_c)


# ============================================================================
# 5. THE STPV SYSTEM
# ============================================================================


@dataclass(eq=False)
class STPVSystem:
    """Full STPV system: absorber + isothermal emitter + PV cell.

    All powers are per unit absorber aperture area [W/m^2].  The absorber
    front face re-radiates into the full hemisphere with eps_abs; the emitter
    (area ratio beta = A_emit / A_abs) radiates toward the PV cell with
    eps_emit; absorber and emitter are isothermal at temperature T.

    Attributes
    ----------
    absorber, emitter : Emissivity
        Spectral/angular emissivity models (TMM-based or analytic).
    concentration : float
        Solar concentration N_s in suns (1 sun = 1 kW/m^2).
    beta : float
        Emitter-to-absorber area ratio A_emit / A_abs.
    cell : PVCell
        PV cell parameters.
    theta_c : float or None
        Half-angle of the incident cone [rad].  If None, it follows from
        etendue conservation: sin(theta_c) = sqrt(N_s) * sin(theta_sun).
    concentrator_efficiency : float
        Optical throughput of the concentrator (0-1) multiplying P_sol.
    optical_filter : DichroicFilter or None
        Optional cold-side shortpass filter.  Its transmittance weights the
        emitter's cold-side channel: reflected sub-bandgap light is recycled
        to the emitter (not a net loss) and never reaches the cell.
    view_factor : float
        Geometric fraction (0-1) of emitter radiation that actually reaches the
        cell, from the emitter/filter/cell sizes and spacings.  Multiplies the
        cell's intrinsic cavity_eff to give the effective cavity efficiency, so
        wider gaps (more spillage) lower CE, the photocurrent, and eta_pvc.
        Left at 1.0 for the base radiative model (no geometric spillage).
    """
    absorber: Emissivity
    emitter: Emissivity
    concentration: float = 500.0
    beta: float = 1.0
    cell: PVCell = field(default_factory=PVCell)
    theta_c: Optional[float] = None
    concentrator_efficiency: float = 1.0
    optical_filter: Optional[DichroicFilter] = None
    view_factor: float = 1.0

    @property
    def effective_cavity_eff(self) -> float:
        """Effective cavity efficiency CE = intrinsic cavity_eff x view_factor."""
        return self.cell.cavity_eff * self.view_factor

    def __post_init__(self) -> None:
        lam = LAM_GRID
        if self.theta_c is not None:
            self.theta_c_eff: float = float(self.theta_c)
        else:
            self.theta_c_eff = float(
                np.arcsin(min(1.0, np.sqrt(self.concentration) * np.sin(THETA_SUN))))
        # Precomputed spectral tables on the master wavelength grid.
        self._lam: np.ndarray = lam
        self._eps_abs_hemi: np.ndarray = self.absorber.hemispherical(lam)
        self._eps_emit_hemi: np.ndarray = self.emitter.hemispherical(lam)
        # Cold-side filter transmittance (all ones if no/disabled filter); the
        # emitter's effective emissivity toward the cell is weighted by it.
        self._filter_tau: np.ndarray = (
            self.optical_filter.transmittance(lam)
            if self.optical_filter is not None else np.ones_like(lam))
        self._eps_emit_cold: np.ndarray = self._eps_emit_hemi * self._filter_tau
        eps_abs_cone = self.absorber.cone_average(lam, self.theta_c_eff)
        self._p_sol: float = self.concentrator_efficiency * self.concentration \
            * float(simpson(eps_abs_cone * am15_spectral_irradiance(lam), x=lam))

    # ------------------------------------------------------------------ power
    def p_sol(self) -> float:
        """Absorbed concentrated solar power P_sol [W/m^2] (T-independent)."""
        return self._p_sol

    def p_abs(self, temperature: float) -> float:
        """Power re-radiated by the absorber front face at T [W/m^2]."""
        radiance = planck_spectral_radiance(self._lam, temperature)
        return np.pi * float(simpson(self._eps_abs_hemi * radiance, x=self._lam))

    def p_emit(self, temperature: float) -> float:
        """Net power radiated by the emitter toward the PV at T [W/m^2 emitter].

        Weighted by the cold-side filter transmittance: reflected sub-bandgap
        light is recycled to the emitter and therefore excluded from this net
        radiative loss.  With no filter this is the full hemispherical emission.
        """
        radiance = planck_spectral_radiance(self._lam, temperature)
        return np.pi * float(simpson(self._eps_emit_cold * radiance, x=self._lam))

    def net_power(self, temperature: float) -> float:
        """Net heating power P_sol - P_abs(T) - beta * P_emit(T) [W/m^2].

        Positive => the module heats up; negative => it cools down; the root
        is the thermal-equilibrium operating temperature.
        """
        return (self._p_sol - self.p_abs(temperature)
                - self.beta * self.p_emit(temperature))

    # ------------------------------------------------------------ equilibrium
    def solve_equilibrium(self, t_lo: float = T_MIN, t_hi: float = T_MAX) -> float:
        """Solve net_power(T) = 0 for the equilibrium temperature T_eq [K].

        Uses scipy.optimize.brentq on [t_lo, t_hi] after verifying that the
        bracket contains a sign change.

        Raises
        ------
        EquilibriumError
            If the system cannot reach equilibrium inside the bounds (either
            it radiates faster than it absorbs even at t_lo, or it would keep
            heating past t_hi and the tungsten module would melt).
        """
        f_lo = self.net_power(t_lo)
        f_hi = self.net_power(t_hi)
        if f_lo < 0.0:
            raise EquilibriumError(
                f"Radiative losses exceed absorbed power even at {t_lo:.0f} K "
                f"(net = {f_lo:.3g} W/m^2): concentration N_s = "
                f"{self.concentration:g} is too low for this design.")
        if f_hi > 0.0:
            raise EquilibriumError(
                f"No equilibrium below {t_hi:.0f} K (net at bound = "
                f"{f_hi:.3g} W/m^2): the module would keep heating and melt "
                f"(tungsten m.p. {T_MELT_W:.0f} K). Reduce N_s or increase beta.")
        return float(brentq(self.net_power, t_lo, t_hi, xtol=0.01))

    # ------------------------------------------------------------- efficiency
    def _emitter_band_integrals(self, temperature: float,
                                lam_g: float) -> Tuple[float, float]:
        """Hemispherical emitter integrals used by SE and the photocurrent.

        Returns
        -------
        (total_power, above_gap_photon_flux) :
            total_power [W/m^2] = pi int eps_emit tau_filter I_BB dlam;
            photon flux [1/(m^2 s)] of emitted photons reaching the cell with
            lam <= lam_g (also weighted by the filter transmittance).
        """
        lam = self._lam
        spectral_power = np.pi * self._eps_emit_cold \
            * planck_spectral_radiance(lam, temperature)
        total = float(simpson(spectral_power, x=lam))
        in_band = np.where(lam <= lam_g, spectral_power * lam / (H * C), 0.0)
        flux = float(simpson(in_band, x=lam))
        return total, flux

    def spectral_efficiency(self, temperature: float) -> float:
        """Spectral efficiency SE of the emitted spectrum w.r.t. the bandgap.

        SE = E_g * Phi_above_gap / P_emit: the fraction of emitted power that
        remains after sub-bandgap photons are discarded and above-bandgap
        photons are thermalized down to E_g.
        """
        total, flux = self._emitter_band_integrals(
            temperature, self.cell.bandgap_wavelength)
        return self.cell.bandgap_j * flux / total if total > 0.0 else 0.0

    def photocurrent_density(self, temperature: float) -> float:
        """Photocurrent density J_L [A/m^2 of cell], J_L = q IQE CE Phi.

        CE here is the effective cavity efficiency (intrinsic cavity_eff times
        the geometric view factor), so gap-driven spillage reduces J_L.
        """
        _, flux = self._emitter_band_integrals(
            temperature, self.cell.bandgap_wavelength)
        return Q * self.cell.iqe * self.effective_cavity_eff * flux

    def voltage_factor(self, j_l: float) -> float:
        """Voltage factor VF = (k_B T_c / (q V_g)) ln(I_L / I_0 + 1).

        Implemented exactly per INSTRUCTIONS.md with current densities;
        clipped to [0, 1] since physically V_oc cannot exceed V_g.
        """
        vf = (KB * self.cell.cell_temp / (Q * self.cell.bandgap_ev)) \
            * np.log1p(j_l / self.cell.effective_dark_current)
        return float(np.clip(vf, 0.0, 1.0))

    def performance(self, temperature: float) -> Dict[str, float]:
        """Full efficiency breakdown of the system at temperature T.

        Returns a dict with the power terms, the five PV factors
        (SE, IQE, VF, FF, CE), eta_int, eta_pvc, and eta_stpv.
        """
        p_sol = self._p_sol
        p_abs = self.p_abs(temperature)
        p_emit = self.p_emit(temperature)
        eta_int = 1.0 - p_abs / p_sol if p_sol > 0.0 else 0.0
        se = self.spectral_efficiency(temperature)
        j_l = self.photocurrent_density(temperature)
        vf = self.voltage_factor(j_l)
        ce = self.effective_cavity_eff
        eta_pvc = se * self.cell.iqe * vf * self.cell.fill_factor * ce
        return {
            "T": temperature, "P_sol": p_sol, "P_abs": p_abs, "P_emit": p_emit,
            "eta_int": eta_int, "SE": se, "IQE": self.cell.iqe, "VF": vf,
            "FF": self.cell.fill_factor, "CE": ce, "view_factor": self.view_factor,
            "J_L": j_l, "eta_pvc": eta_pvc, "eta_stpv": eta_int * eta_pvc,
        }


# ============================================================================
# 6. PARAMETRIC SWEEPS AND PLOTTING
# ============================================================================

plt.rcParams.update({
    "figure.figsize": (9.0, 5.6),
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 11,
    "axes.titlesize": 12,
    "lines.linewidth": 2.0,
    "savefig.dpi": 160,
})


@dataclass(eq=False)
class Sweeper:
    """Runs the five required parametric studies on a base STPV system.

    Each `plot_*` method returns a matplotlib Figure and saves it to
    `output_dir` with a descriptive filename before it is shown.
    """
    base: STPVSystem
    output_dir: Path = _PROJECT_DIR

    # ------------------------------------------------------------------ utils
    def _finish(self, fig: plt.Figure, filename: str) -> plt.Figure:
        """Apply tight_layout, save the figure to disk, and report the path."""
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path)
        print(f"  [plot] saved {path.name}")
        return fig

    def _solve_or_nan(self, system: STPVSystem) -> float:
        """Equilibrium temperature, or NaN if none exists within bounds."""
        try:
            return system.solve_equilibrium()
        except EquilibriumError:
            return float("nan")

    # ------------------------------------------------- 1. equilibrium proof
    def plot_equilibrium(self) -> plt.Figure:
        """Net power vs temperature, visually proving the brentq root."""
        temps = np.linspace(300.0, T_MAX, 161)
        p_sol = self.base.p_sol()
        losses = np.array(
            [self.base.p_abs(t) + self.base.beta * self.base.p_emit(t)
             for t in temps])
        net = p_sol - losses
        t_eq = self.base.solve_equilibrium()

        fig, ax = plt.subplots()
        ax.plot(temps, net / 1e3, color="tab:blue",
                label=r"Net power $P_{sol} - P_{abs} - \beta P_{emit}$")
        ax.plot(temps, np.full_like(temps, p_sol) / 1e3, "--",
                color="tab:orange", lw=1.4, label=r"$P_{sol}$ (input)")
        ax.plot(temps, losses / 1e3, "--", color="tab:red", lw=1.4,
                label=r"$P_{abs} + \beta P_{emit}$ (radiative losses)")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.plot([t_eq], [0.0], "o", ms=9, color="tab:green", zorder=5,
                label=f"Equilibrium: $T_{{eq}}$ = {t_eq:.0f} K")
        ax.set_xlabel("Module temperature $T$ [K]")
        ax.set_ylabel(r"Power density [kW/m$^2$ of absorber]")
        ax.set_title(f"Thermal equilibrium ($N_s$ = {self.base.concentration:g} "
                     f"suns, $\\beta$ = {self.base.beta:g})")
        ax.legend(loc="best")
        return self._finish(fig, "fig1_thermal_equilibrium.png")

    # ------------------------------------------------- 2. concentration sweep
    def plot_concentration_sweep(self) -> plt.Figure:
        """N_s from 1 to 10,000 suns vs T_eq and eta_STPV."""
        ns_vals = np.geomspace(1.0, 1.0e4, 49)
        t_eq = np.full_like(ns_vals, np.nan)
        eta = np.full_like(ns_vals, np.nan)
        for i, ns in enumerate(ns_vals):
            system = dc_replace(self.base, concentration=float(ns))
            t = self._solve_or_nan(system)
            if np.isfinite(t):
                t_eq[i] = t
                eta[i] = system.performance(t)["eta_stpv"]
        n_fail = int(np.sum(~np.isfinite(t_eq)))
        if n_fail:
            print(f"  [sweep] concentration: {n_fail}/{ns_vals.size} points "
                  f"have no equilibrium below {T_MAX:.0f} K (module would melt)")

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        l1, = ax1.semilogx(ns_vals, 100.0 * eta, color="tab:blue",
                           label=r"$\eta_{STPV}$")
        l2, = ax2.semilogx(ns_vals, t_eq, color="tab:red",
                           label=r"$T_{eq}$")
        ax2.axhline(T_MELT_W, color="tab:red", ls=":", lw=1.2)
        ax2.text(ns_vals[-1], T_MELT_W - 110, "tungsten melting point",
                 color="tab:red", fontsize=9, ha="right")
        ax1.set_xlabel(r"Solar concentration $N_s$ [suns]")
        ax1.set_ylabel(r"Overall efficiency $\eta_{STPV}$ [%]",
                       color="tab:blue")
        ax2.set_ylabel(r"Equilibrium temperature $T_{eq}$ [K]",
                       color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.grid(False)
        ax1.set_title(r"Concentration sweep: $\eta_{STPV}$ and $T_{eq}$ vs $N_s$"
                      f" ($\\beta$ = {self.base.beta:g})")
        ax1.legend(handles=[l1, l2], loc="upper left")
        return self._finish(fig, "fig2_concentration_sweep.png")

    # --------------------------------------------------- 3. area-ratio sweep
    def plot_area_ratio_sweep(self) -> plt.Figure:
        """beta from 0.1 to 10 vs eta_STPV (and T_eq) for geometry sizing."""
        betas = np.geomspace(0.1, 10.0, 49)
        t_eq = np.full_like(betas, np.nan)
        eta = np.full_like(betas, np.nan)
        for i, b in enumerate(betas):
            system = dc_replace(self.base, beta=float(b))
            t = self._solve_or_nan(system)
            if np.isfinite(t):
                t_eq[i] = t
                eta[i] = system.performance(t)["eta_stpv"]
        i_best = int(np.nanargmax(eta))

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        l1, = ax1.semilogx(betas, 100.0 * eta, color="tab:blue",
                           label=r"$\eta_{STPV}$")
        l2, = ax2.semilogx(betas, t_eq, color="tab:red", label=r"$T_{eq}$")
        l3, = ax1.plot([betas[i_best]], [100.0 * eta[i_best]], "o", ms=9,
                       color="tab:green", zorder=5,
                       label=(f"Optimum: $\\beta$ = {betas[i_best]:.2f}, "
                              f"$\\eta$ = {100 * eta[i_best]:.2f}%"))
        ax1.set_xlabel(r"Area ratio $\beta = A_{emit} / A_{abs}$")
        ax1.set_ylabel(r"Overall efficiency $\eta_{STPV}$ [%]",
                       color="tab:blue")
        ax2.set_ylabel(r"Equilibrium temperature $T_{eq}$ [K]",
                       color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.grid(False)
        ax1.set_title(r"Area-ratio sweep: $\eta_{STPV}$ vs $\beta$"
                      f" ($N_s$ = {self.base.concentration:g} suns)")
        ax1.legend(handles=[l1, l2, l3], loc="best")
        return self._finish(fig, "fig3_area_ratio_sweep.png")

    # -------------------------------------------------- 4. dark-current sweep
    def plot_dark_current_sweep(self) -> plt.Figure:
        """J_0 (log scale) vs VF and eta_pvc; marks the degradation onset."""
        t_eq = self.base.solve_equilibrium()
        j0_vals = np.geomspace(1.0e-12, 1.0e2, 57)   # A/m^2
        vf = np.empty_like(j0_vals)
        eta_pvc = np.empty_like(j0_vals)
        for i, j0 in enumerate(j0_vals):
            # Clear rated Voc/Jsc so dark_current drives VF for this sweep.
            system = dc_replace(self.base, cell=dc_replace(
                self.base.cell, dark_current=j0, rated_voc=None, rated_jsc=None))
            perf = system.performance(t_eq)
            vf[i] = perf["VF"]
            eta_pvc[i] = perf["eta_pvc"]
        # VF falls log-linearly in J_0 (no plateau), so define the maximum
        # allowable dark current as the point where eta_pvc drops 10% below
        # its value for the default (realistic) cell.
        j0_ref = self.base.cell.effective_dark_current
        eta_ref = np.interp(np.log10(j0_ref), np.log10(j0_vals), eta_pvc)
        below = np.nonzero(eta_pvc < 0.9 * eta_ref)[0]
        j0_max = j0_vals[below[0]] if below.size else j0_vals[-1]

        fig, ax = plt.subplots()
        ax.semilogx(j0_vals, vf, color="tab:blue", label="Voltage factor VF")
        ax.semilogx(j0_vals, eta_pvc, color="tab:orange",
                    label=r"PV-cell efficiency $\eta_{pvc}$")
        ax.axvline(j0_ref, color="gray", ls=":", lw=1.2,
                   label=(f"Default Si cell ($J_0$ = {j0_ref:.0e} A/m$^2$)"))
        ax.axvline(j0_max, color="tab:red", ls="--", lw=1.4,
                   label=(r"10% degradation vs default at $J_0 \approx$ "
                          f"{j0_max:.1e} A/m$^2$"))
        ax.set_xlabel(r"Dark current density $J_0$ [A/m$^2$]"
                      r"  (1 A/m$^2$ = $10^{-4}$ A/cm$^2$)")
        ax.set_ylabel("Factor value [-]")
        ax.set_title(f"Dark-current sweep at $T_{{eq}}$ = {t_eq:.0f} K "
                     f"($J_L$ = {self.base.photocurrent_density(t_eq):.0f} "
                     r"A/m$^2$)")
        ax.legend(loc="best")
        return self._finish(fig, "fig4_dark_current_sweep.png")

    # ------------------------------------------------------ 5. bandgap sweep
    def plot_bandgap_sweep(self) -> plt.Figure:
        """PV bandgap 0.5-2.0 eV vs eta_STPV: blackbody vs narrowband emitter."""
        bandgaps = np.linspace(0.5, 2.0, 61)
        emitters: Sequence[Emissivity] = (
            gray_emissivity(1.0, "Idealized blackbody emitter"),
            step_band_emissivity(
                0.85, 0.05, 1.5e-6, 2.4e-6,
                "Narrowband emitter (0.85 in 1.5-2.4 um)"),
        )
        fig, ax = plt.subplots()
        for emitter, color in zip(emitters, ("tab:red", "tab:blue")):
            system = dc_replace(self.base, emitter=emitter)
            t_eq = self._solve_or_nan(system)
            if not np.isfinite(t_eq):
                print(f"  [sweep] bandgap: no equilibrium for '{emitter.name}'")
                continue
            eta = np.empty_like(bandgaps)
            for i, eg in enumerate(bandgaps):
                sys_i = dc_replace(
                    system, cell=dc_replace(system.cell, bandgap_ev=float(eg)))
                eta[i] = sys_i.performance(t_eq)["eta_stpv"]
            i_best = int(np.argmax(eta))
            ax.plot(bandgaps, 100.0 * eta, color=color,
                    label=f"{emitter.name} ($T_{{eq}}$ = {t_eq:.0f} K)")
            ax.plot([bandgaps[i_best]], [100.0 * eta[i_best]], "o", ms=7,
                    color=color)
            print(f"  [sweep] bandgap: '{emitter.name}': T_eq = {t_eq:.0f} K, "
                  f"best E_g = {bandgaps[i_best]:.2f} eV "
                  f"(eta_STPV = {100 * eta[i_best]:.2f}%)")
        ax.axvline(1.12, color="gray", ls=":", lw=1.2)
        ax.text(1.13, ax.get_ylim()[1] * 0.93, "Si (1.12 eV)", color="gray",
                fontsize=9)
        ax.set_xlabel(r"PV bandgap $V_g$ [eV]")
        ax.set_ylabel(r"Overall efficiency $\eta_{STPV}$ [%]")
        ax.set_title(r"Bandgap sweep: blackbody vs narrowband emitter"
                     f" ($N_s$ = {self.base.concentration:g} suns, "
                     f"$\\beta$ = {self.base.beta:g})")
        ax.legend(loc="best")
        return self._finish(fig, "fig5_bandgap_sweep.png")

    def run_all(self) -> List[plt.Figure]:
        """Generate, save, and return all five required sweep figures."""
        return [
            self.plot_equilibrium(),
            self.plot_concentration_sweep(),
            self.plot_area_ratio_sweep(),
            self.plot_dark_current_sweep(),
            self.plot_bandgap_sweep(),
        ]


# ============================================================================
# 7. VALIDATION AND MAIN DRIVER
# ============================================================================

def run_validation() -> None:
    """Fail-fast numerical checks of the integration machinery.

    1. A perfect blackbody's P_emit reproduces sigma T^4.
    2. A perfect absorber collects exactly N_s * 1 kW/m^2 of AM1.5 sunlight.
    3. The THEORY.md step-emitter example at 1500 K matches an independent
       adaptive-quadrature (scipy.integrate.quad) evaluation.
    4. At equilibrium, eta_int = 1 - P_abs/P_sol equals beta*P_emit/P_sol.
    """
    print("Running validation checks...")
    perfect = gray_emissivity(1.0)

    # 1: Stefan-Boltzmann.
    system = STPVSystem(absorber=perfect, emitter=perfect, concentration=100.0)
    p_bb = system.p_emit(1500.0)
    exact = SIGMA * 1500.0**4
    err = abs(p_bb / exact - 1.0)
    assert err < 2e-3, f"Blackbody integral error {err:.2e}"
    print(f"  [ok] blackbody P_emit(1500 K) = {p_bb:.4g} W/m^2 "
          f"vs sigma*T^4 = {exact:.4g} (err {err:.1e})")

    # 2: solar normalization.
    p_sol = system.p_sol()
    err = abs(p_sol / (100.0 * ONE_SUN) - 1.0)
    assert err < 1e-2, f"AM1.5 normalization error {err:.2e}"
    print(f"  [ok] perfect-absorber P_sol(100 suns) = {p_sol:.4g} W/m^2 "
          f"vs 1e5 (err {err:.1e})")

    # 3: THEORY.md step-emitter example, cross-checked with adaptive quad.
    step = step_band_emissivity(0.85, 0.05, 1.5e-6, 2.4e-6)
    system_step = STPVSystem(absorber=perfect, emitter=step)
    p_step = system_step.p_emit(1500.0)
    band, _ = quad(lambda l: np.pi * float(
        planck_spectral_radiance(np.array([l]), 1500.0)[0]), 1.5e-6, 2.4e-6,
        limit=200)
    expected = 0.85 * band + 0.05 * (exact - band)
    err = abs(p_step / expected - 1.0)
    # The step emitter is discontinuous at the band edges, which fall between
    # fixed grid points, so allow a slightly looser (still sub-1%) tolerance.
    assert err < 1e-2, f"Step-emitter integral error {err:.2e}"
    print(f"  [ok] step emitter P_emit(1500 K) = {p_step:.4g} W/m^2 "
          f"vs quad-based {expected:.4g} (err {err:.1e})")

    # 4: energy-balance consistency of the two eta_int formulations.
    t_eq = system_step.solve_equilibrium()
    p = system_step.performance(t_eq)
    alt = system_step.beta * p["P_emit"] / p["P_sol"]
    err = abs(p["eta_int"] - alt)
    assert err < 1e-3, f"eta_int consistency error {err:.2e}"
    print(f"  [ok] at T_eq = {t_eq:.1f} K: eta_int = {p['eta_int']:.4f} "
          f"(1 - P_abs/P_sol) = {alt:.4f} (beta*P_emit/P_sol)")
    print("All validation checks passed.\n")


def build_default_system(verbose: bool = True) -> STPVSystem:
    """Construct the STPV system from the USER INPUT PARAMETERS block.

    Wires in the concentrator optical efficiency (scales P_sol), the
    absorber/emitter material + geometry (emissivity models and nominal beta),
    the optional cold-side filter, the PV-cell configuration, and the geometric
    view factor (from the spacings/sizes) that sets the effective cavity CE.
    """
    absorber, emitter = ABSORBER_EMITTER.build(verbose=verbose)
    return STPVSystem(
        absorber=absorber, emitter=emitter,
        concentration=NS_DEFAULT, beta=ABSORBER_EMITTER.nominal_beta,
        cell=PV_CELL, concentrator_efficiency=CONCENTRATOR.active.efficiency,
        optical_filter=DICHROIC_FILTER, view_factor=SYSTEM_VIEW_FACTOR)


def main() -> None:
    """Validate, solve the default system, and produce the 5 sweep figures."""
    t_start = time.perf_counter()
    print("=" * 72)
    print("STPV system model -- thermal equilibrium and efficiency analysis")
    print("=" * 72)

    run_validation()

    print("Building system from USER INPUT PARAMETERS "
          "(TMM emissivities computed once)...")
    system = build_default_system()

    print("\nSystem configuration")
    print("-" * 72)
    CONCENTRATOR.print_summary()
    active = CONCENTRATOR.active
    print(f"      geometric conc.      : {GEOMETRIC_CONCENTRATION:,.0f} suns "
          f"(A_aper / A_abs); x eff -> "
          f"{GEOMETRIC_CONCENTRATION * active.efficiency:,.0f} suns delivered")
    ABSORBER_EMITTER.print_summary()
    print(f"      absorber surface     : {system.absorber.name}")
    print(f"      emitter surface      : {system.emitter.name}")
    DICHROIC_FILTER.print_summary()
    system.cell.print_summary()
    if USE_VIEW_FACTOR_CE:
        path = ("emitter->filter->cell" if DICHROIC_FILTER.enabled
                else "emitter->cell")
        print(f"      view factor ({path}): {system.view_factor:.3f} "
              f"-> effective CE = {system.effective_cavity_eff:.3f} "
              f"(intrinsic {system.cell.cavity_eff:.2f})")
    else:
        print(f"      view factor          : disabled (CE = "
              f"{system.effective_cavity_eff:.2f})")
    print("-" * 72)
    ns_src = ("manual override" if NS_OVERRIDE is not None
              else "derived from concentrator/absorber sizes")
    print(f"  N_s = {system.concentration:,.0f} suns ({ns_src}), "
          f"concentrator eff = {system.concentrator_efficiency:.2f}, "
          f"beta = {system.beta:g},\n"
          f"  theta_c = {np.rad2deg(system.theta_c_eff):.2f} deg (etendue). "
          f"N_s is varied independently in the concentration sweep.\n")

    t_eq = system.solve_equilibrium()
    perf = system.performance(t_eq)
    print("Default operating point:")
    print(f"  T_eq     = {t_eq:8.1f} K")
    print(f"  P_sol    = {perf['P_sol']:10.1f} W/m^2")
    print(f"  P_abs    = {perf['P_abs']:10.1f} W/m^2 (re-radiated)")
    print(f"  P_emit   = {perf['P_emit']:10.1f} W/m^2 (to PV, x beta = "
          f"{system.beta:g})")
    print(f"  eta_int  = {100 * perf['eta_int']:7.2f} %")
    print(f"  SE = {perf['SE']:.3f}, IQE = {perf['IQE']:.2f}, "
          f"VF = {perf['VF']:.3f}, FF = {perf['FF']:.2f}, CE = {perf['CE']:.2f}"
          f"  (J_L = {perf['J_L']:.1f} A/m^2)")
    print(f"  eta_pvc  = {100 * perf['eta_pvc']:7.2f} %")
    print(f"  eta_STPV = {100 * perf['eta_stpv']:7.2f} %\n")

    print("Generating parametric sweeps...")
    sweeper = Sweeper(system)
    sweeper.run_all()

    print(f"\nDone in {time.perf_counter() - t_start:.1f} s. "
          "Displaying figures...")
    plt.show()


if __name__ == "__main__":
    main()
