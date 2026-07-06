# Solar Thermophotovoltaic (STPV) Python Model Theory

This document outlines the step-by-step instructions for building a numerical Python model to calculate and optimize the efficiency of an STPV system.

## Step 1: Define System Architecture (Optical Design)

Your model should allow for the selection and parametrization of the following five core components:

1. **Concentrator ($C$):** * Options: Parabolic dish (with a possible second stage) OR Fresnel lens.

2. **Absorber ($\eta_a$):** * Options: Selective absorber (used for 0 to a few hundred suns) OR Blackbody absorber (used for high hundreds/thousands of suns).

3. **Emitter:** * Options: Step-function edge emitters OR Narrowband emitters. Material options (same as absorber): Tungsten, tantalum, graphite, silicon carbide, stainless steel, etc. Size options (same as absorber): Rectangular prism, 1 to 5 $cm^2$ face area, 0.1 to 1 $cm$ thickness.

4. **PV Cell:** * Options: Si cell (Single junction, 1.12 eV). 1 to 5 $cm^2$ active face area.

5. **Thermal Management:** * Options: Heatsink designs.

## Step 2: Setup Core Variables and Power Integrals

Define the foundational power equations that will dictate energy transfer in the system.

### 2.1 Power Equations

Set up numerical integration in Python (e.g., using scipy.integrate) for the following power terms. Note that spatial integrals represent the area of light, and multiplying by intensity gives the power.

1. **Power of Incident Sunlight ($P_{sol}$)**


$$P_{sol} = \int_{0}^{2\pi} d\phi \int_{0}^{\theta_c} \sin(\theta)\cos(\theta) d\theta \int_{0}^{\infty} \varepsilon_{abs}(\lambda, \phi, \theta) I_{AM1.5}(\lambda, T) d\lambda$$

- *Note:* The spatial integral represents the conical angle of incident sunlight. $I_{AM1.5}$ is the incident sunlight intensity (1 sun = $1 kW/m^2$), multiplied by the concentration factor $N_s$.

- $\varepsilon_{abs}$: Transfer Matrix Method (TMM) absorber properties.

2. **Re-radiated Light/Absorbed Power ($P_{abs}$)**


$$P_{abs} = \int_{0}^{2\pi} d\phi \int_{0}^{\pi/2} \sin(\theta)\cos(\theta) d\theta \int_{0}^{\infty} \varepsilon_{abs}(\lambda, \phi, \theta) I_{BB}(\lambda, T) d\lambda$$

- *Note:* $I_{BB}$ is Planck's Law for the absorber as a blackbody. Ideal $= \sigma T^4$, Accurate $= \varepsilon_{abs} \sigma T^4$.

3. **Output Emittance ($P_{emit}$)**


$$P_{emit} = \int_{0}^{2\pi} d\phi \int_{0}^{\pi/2} \sin(\theta)\cos(\theta) d\theta \int_{0}^{\infty} \varepsilon_{emit}(\lambda, \phi, \theta) I_{BB}(\lambda, T) d\lambda$$

- *Note:* $\varepsilon_{emit}$ represents TMM output emittance. Consider selectivity.

- *Example formulation for numerical test:*

$$P_{emit} = \int_{1.5\mu m}^{2.4\mu m} 0.85 I_{BB}(\lambda, 1500K) d\lambda + \int_{rest} 0.05 I_{BB}(\lambda, 1500K) d\lambda$$

## Step 3: Solve for Thermal Equilibrium

Before calculating final efficiencies, the model must find the operating temperature where the system reaches thermal equilibrium.

1. **Define Area Ratio ($\beta$):**

$$\beta = \frac{A_{emit}}{A_{abs}}$$

2. **Define the Equilibrium Condition:**

$$P_{sol} - P_{abs} = \beta P_{emit}$$

- *Logic:* If Left > Right, more energy is entering than leaving, so the temperature is rising (and vice versa).

3. **Python Implementation**: * Create a function: net_power(T) = P_sol - P_abs(T) - beta * P_emit(T)

- Iterate (using a root-finding algorithm like scipy.optimize.fsolve) to find the temperature $T$ at which net_power(T) == 0.

- Plot $P_{sol} - P_{abs} - \beta P_{emit}$ against $T$ to visualize the equilibrium point.

## Step 4: Compute Sub-System and Overall Efficiencies

Once the equilibrium temperature is found, evaluate the efficiency.

### 4.1 Intermediate Efficiency ($\eta_{int}$)

Evaluated at the equilibrium temperature:


$$\eta_{int} = 1 - \frac{P_{abs}}{P_{sol}}$$


*(Alternatively expressed via the notes as $\eta_{int} = \beta \frac{P_{emit}}{P_{sol}}$)*

### 4.2 PV Cell Efficiency ($\eta_{pvc}$)

Break this down into 5 multiplicative factors:


$$\eta_{pvc} = SE \cdot IQE \cdot VF \cdot FF \cdot CE$$

- **SE (Spectral Efficiency):** Accounts for spectral losses. Sweep TMM parameters to find the optimal SE.

- **IQE (Internal Quantum Efficiency):** Usually found in literature or estimated.

- **VF (Voltage Factor):**


$$VF = \frac{k_B T_c}{q V_g} \ln\left(\frac{I_L}{I_0} + 1\right)$$


*Sweep to find ideal $I_0$ (dark current) to see how good the cell needs to be.*

- **FF (Fill Factor):** Given by the manufacturer.

- **CE (Cavity Efficiency):** Found or estimated.

### 4.3 Overall STPV Efficiency ($\eta_{STPV}$)

$$\eta_{STPV} = \eta_{int} \cdot \eta_{pvc}$$


*(Note: If optical concentration stages are separate, it may be formulated as $\eta_{STPV} = C \cdot \eta_{int} \cdot \eta_{pvc}$)*

## Step 5: Implement Parametric Sweeps

Your Python model should be built to loop through and plot the following parameter sweeps:

1. **Optical / Concentration Sweeps:**

- **Concentration ($N_s$):** Sweep suns to find the minimum needed concentration for efficiency, and observe its effect on temperature.

- **Wavelength:** Look for chromatic aberration effects.

- **Area Ratio ($\beta$):** Sweep to help determine the physical size constraints of the system.

2. **Material / TMM Sweeps:**

- **Absorber/Emitter Efficiency:** Sweep module temperature, absorption, and self-radiation.

- **TMM Absorber Properties:** Sweep to observe changes in $\varepsilon_{abs}$.

- **TMM Emitter Properties:** Sweep to observe changes in $\varepsilon_{emit}$ and $SE$.

3. **PV Cell Sweeps:**

- **Dark Current ($I_0$):** Sweep within realistic limits to find the maximum $I_0$ you can have before performance degrades.

- **Cell Temperature / Cavity:** Sweep to influence material choices.

- **Other parameters:** Bandgap, Fill Factor, IQE, Spectral Efficiency, $J_{sc}$, $V_{oc}/I_{sc}$.