#!/bin/bash
from openmm.app import *
from openmm import *
import os
from sys import stdout, exit, stderr
import numpy as np
import sys
import mdtraj as md
import openmm.unit as unit

xlength = 90.2632491/10
ylength = 90.2632491/10
zlength = 118.618231/10

psf = CharmmPsfFile('../../1-min/8y2d_ions_da.psf')
#psf = PDBFile('last_frame_100ns.pdb')
psf.setBox(xlength,ylength,zlength)
cor = CharmmCrdFile('../../1-min/8y2d_ions_da_minimized.crd')

params = CharmmParameterSet('../../toppar/top_all36_prot.rtf',
'../../toppar/par_all36m_prot.prm',
'../../toppar/top_all36_na.rtf',
'../../toppar/par_all36_na.prm',
'../../toppar/top_all36_carb.rtf',
'../../toppar/par_all36_carb.prm',
'../../toppar/top_all36_lipid.rtf',
'../../toppar/par_all36_lipid.prm',
'../../toppar/top_all36_cgenff.rtf',
'../../toppar/par_all36_cgenff.prm',
'../../toppar/toppar_all36_moreions.str',
'../../toppar/top_interface.rtf',
'../../toppar/par_interface.prm',
'../../toppar/toppar_all36_nano_lig.str',
'../../toppar/toppar_all36_nano_lig_patch.str',
'../../toppar/toppar_all36_synthetic_polymer.str',
'../../toppar/toppar_all36_synthetic_polymer_patch.str',
'../../toppar/toppar_all36_polymer_solvent.str',
'../../toppar/toppar_water_ions.str',
'../../toppar/toppar_dum_noble_gases.str',
'../../toppar/toppar_ions_won.str',
'../../toppar/cam.str',
'../../toppar/toppar_all36_prot_arg0.str',
'../../toppar/toppar_all36_prot_c36m_d_aminoacids.str',
'../../toppar/toppar_all36_prot_fluoro_alkanes.str',
'../../toppar/toppar_all36_prot_heme.str',
'../../toppar/toppar_all36_prot_na_combined.str',
'../../toppar/toppar_all36_prot_retinol.str',
'../../toppar/toppar_all36_prot_model.str',
'../../toppar/toppar_all36_prot_modify_res.str',
'../../toppar/toppar_all36_na_nad_ppi.str',
'../../toppar/toppar_all36_na_rna_modified.str',
'../../toppar/toppar_all36_lipid_sphingo.str',
'../../toppar/toppar_all36_lipid_archaeal.str',
'../../toppar/toppar_all36_lipid_bacterial.str',
'../../toppar/toppar_all36_lipid_cardiolipin.str',
'../../toppar/toppar_all36_lipid_cholesterol.str',
'../../toppar/toppar_all36_lipid_dag.str',
'../../toppar/toppar_all36_lipid_inositol.str',
'../../toppar/toppar_all36_lipid_lnp.str',
'../../toppar/toppar_all36_lipid_lps.str',
'../../toppar/toppar_all36_lipid_mycobacterial.str',
'../../toppar/toppar_all36_lipid_miscellaneous.str',
'../../toppar/toppar_all36_lipid_model.str',
'../../toppar/toppar_all36_lipid_prot.str',
'../../toppar/toppar_all36_lipid_tag.str',
'../../toppar/toppar_all36_lipid_yeast.str',
'../../toppar/toppar_all36_lipid_hmmm.str',
'../../toppar/toppar_all36_lipid_detergent.str',
'../../toppar/toppar_all36_lipid_ether.str',
'../../toppar/toppar_all36_lipid_oxidized.str',
'../../toppar/toppar_all36_carb_glycolipid.str',
'../../toppar/toppar_all36_carb_glycopeptide.str',
'../../toppar/toppar_all36_carb_imlab.str',
'../../toppar/toppar_all36_label_spin.str',
'../../toppar/toppar_all36_label_fluorophore.str',
'../../ldp/ldp.rtf',
'../../ldp/ldp.prm')

tstep= 50000000
# === Create System ===
system = psf.createSystem(params, nonbondedMethod=PME,
                          nonbondedCutoff=1.2*unit.nanometers,
                          constraints=HBonds,
                          rigidWater=True, ewaldErrorTolerance=0.0005)


# === Load occluded structure and compute target distance ===
occluded = md.load('8y2c.pdb')
top = md.load_psf('../../1-min/8y2d_ions_da.psf')

# === Identify TM1, TM3, TM6, TM8 CA Atoms and Masses ===
tm_residues = list(range(1, 31)) + list(range(128, 172)) + list(range(242, 271)) + list(range(336, 378))

# === Identify TM CA Atoms and Masses from PSF ===
tm_ca_atoms = []
tm_ca_weights = []
for atom in psf.topology.atoms():
    if atom.name == 'CA':
        try:
            resid = int(atom.residue.id)
            if resid in tm_residues:
                tm_ca_atoms.append(atom.index)
                tm_ca_weights.append(atom.element.mass.value_in_unit(unit.dalton))
        except ValueError:
            continue  # skip non-integer residue IDs

# === Compute COM of TM CAs in occluded structure ===
tm_ca_indices_in_occluded = [atom.index for atom in occluded.topology.atoms 
                             if atom.name == 'CA' and atom.residue.resSeq in tm_residues]
com_tm = np.mean(occluded.xyz[0, tm_ca_indices_in_occluded], axis=0)

# === Target distance can be updated here ===
target_distance = 0.0  # or some other value if pulling against another group

# === Add Distance-Based Pulling Force ===
pull_force = CustomCentroidBondForce(2, '0.5 * k * (distance(g1, g2) - target)^2')
pull_force.addGlobalParameter('k', 1000.0)         # kJ/mol/nm^2
pull_force.addGlobalParameter('target', target_distance)
pull_force.addGroup(tm_ca_atoms, tm_ca_weights)
pull_force.addGroup(tm_ca_atoms, tm_ca_weights)  # Using same group as placeholder
pull_force.addBond([0, 1], [])
system.addForce(pull_force)
# === Ion Charge Scaling (Optional) ===
nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
charge_scale = 0.7
for i, atom in enumerate(psf.topology.atoms()):
    if atom.residue.name in ('SOD', 'CLA'):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        nonbonded.setParticleParameters(i, charge * charge_scale, sigma, epsilon)
nonbonded.updateParametersInContext

# === Integrator and Platform ===
integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtoseconds)
integrator.setConstraintTolerance(1e-5)
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'single'}

# === Create Simulation ===
simulation = Simulation(psf.topology, system, integrator, platform, properties)
#simulation.context.setPositions(cor.positions)


nonbonded.updateParametersInContext(simulation.context)

# Load checkpoint if exists (adjust path and args as needed)
checkpoint_path = f'../../2-npt/{sys.argv[4]}/{sys.argv[2]}-{sys.argv[4]}.chk'
try:
    simulation.loadCheckpoint(checkpoint_path)
except Exception as e:
    print(f"Warning: Could not load checkpoint '{checkpoint_path}': {e}")

# Add reporters
simulation.reporters.append(DCDReporter(sys.argv[1]+'_'+sys.argv[3]+'-'+sys.argv[4]+'.dcd', 5000))
simulation.reporters.append(StateDataReporter(sys.argv[1]+'_'+sys.argv[3]+'-'+sys.argv[4]+'.log', 5000, step=True, potentialEnergy=True, temperature=True, volume=True, progress=True, remainingTime=True, speed=True, totalSteps=tstep))
simulation.reporters.append(CheckpointReporter(sys.argv[1]+'_'+sys.argv[3]+'-'+sys.argv[4]+'.chk', 5000))

# Minimize energy
print('Minimizing...')
simulation.minimizeEnergy()

# Equilibrate
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
simulation.step(1000)

# Save first frame PDB
state = simulation.context.getState(getPositions=True)
positions = state.getPositions()
with open('first_frame.pdb', 'w') as f:
    app.PDBFile.writeFile(simulation.topology, positions, f)

# Production run (adjust steps as needed)
print('Starting production...')
for frame in range(50000000):
    simulation.step(1)

