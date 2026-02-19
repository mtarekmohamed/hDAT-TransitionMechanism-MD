

# hDAT-TransitionMechanism-MD

## Overview
This repository contains the molecular dynamics simulation workflows, targeted molecular dynamics (TMD) protocols, and representative structures used to characterize conformational transitions of the **human dopamine transporter (hDAT)**. The scripts were used to generate transition trajectories, sample intermediate conformations, and identify key macrostates reported in the associated study describing the atomistic transport mechanism of hDAT.

The repository provides:

- targeted molecular dynamics simulation drivers  
- biased and conventional MD workflows  
- transition pathway generation tools  
- representative macrostate structures  

---

## Relation to the Associated Study
The scripts and structural data in this repository were used to:

- generate transition trajectories between functional hDAT states  
- characterize ion-coupled conformational changes  
- analyze extracellular and intracellular gating rearrangements  
- identify intermediate macrostates (Macrostate 1, 4, and 5)  
- reconstruct the sequence of structural transitions underlying dopamine transport  

---

## Repository Structure

### `TMD/` — Targeted Molecular Dynamics Workflows

This directory contains OpenMM-based simulation drivers used to generate biased and targeted trajectories for sampling large-scale conformational transitions.

The scripts implement RMSD-based steering, collective-variable restraints, helix-bundle pulling, and conventional MD simulations used to generate structural ensembles analyzed in the study.

---

### **`tmd_rmsd.py` — RMSD-based Targeted Molecular Dynamics**
Implements RMSD-driven steering toward a reference structure using OpenMM `RMSDForce`.

**Key features:**
- Residue-matched Cα mapping between reference and system structures  
- Structure alignment prior to bias construction  
- Harmonic RMSD restraint using `CustomCVForce`  
- Controlled structural transition toward a target conformation  
- Ion charge scaling (SOD/CLA) for electrostatic consistency  
- CUDA-accelerated simulation with checkpoint restart support  

**Role in study:**  
Generates transition trajectories connecting functional states and samples intermediate conformations used for pathway reconstruction.

---

### **`tmd_gate.py` — PLUMED-driven Gate Restraint Simulations**
Runs biased MD simulations using PLUMED collective-variable restraints.

**Key features:**
- Integration of PLUMED bias (`plumed_ic.dat`) via `PlumedForce`  
- Gate-focused collective variable restraints  
- Ion charge scaling (SOD/CLA)  
- Restartable simulations with trajectory and checkpoint output  

**Role in study:**  
Characterizes gating transitions and structural rearrangements associated with dopamine transport.

---

### **`tmd_tm1368.py` — Transmembrane Helix Bundle Pulling**
Applies centroid-based harmonic restraints to selected transmembrane helices.

**Key features:**
- Selection of TM1, TM3, TM6, and TM8 Cα atoms  
- Center-of-mass/centroid distance restraint using `CustomCentroidBondForce`  
- Helix-bundle structural manipulation  
- Ion charge scaling and checkpoint restart  

**Role in study:**  
Probes large-scale helix rearrangements and intracellular opening mechanisms.

---

### **`conventional_scaled.py` — Conventional MD with Scaled Charges**
Performs unbiased molecular dynamics simulations with scaled electrostatic interactions.

**Key features:**
- Charge scaling for sodium (SOD), chloride (CLA), and ligand atoms  
- Langevin dynamics simulation  
- Checkpoint restart and trajectory output  
- Used as reference or relaxation simulations  

**Role in study:**  
Provides equilibrium sampling and baseline structural dynamics.

---

### `Macrostates_Structures/` — Representative Macrostate Structures

This directory contains representative structures corresponding to intermediate macrostates identified from trajectory clustering.

**Included structures:**
- **Macrostate 1** — extracellular reopening intermediate  
- **Macrostate 4** — kinetic bottleneck intermediate  
- **Macrostate 5** — extracellular gate collapse intermediate  

These structures represent centroid conformations used for mechanistic interpretation of transport dynamics.

---

## Simulation Framework

All simulations use:

- OpenMM molecular dynamics engine  
- CHARMM force field parameters  
- Particle Mesh Ewald electrostatics  
- Langevin dynamics integrator  
- CUDA GPU acceleration  

Several workflows support restarting from pre-equilibrated NPT checkpoints.

---


