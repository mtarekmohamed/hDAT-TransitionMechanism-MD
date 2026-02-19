#!/bin/env python
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit, stderr
import numpy as np
import sys
import mdtraj as md  
import openmm.unit as unit

xlength = 90.2632491/10
ylength = 90.2632491/10
zlength = 118.618231/10

psf = CharmmPsfFile('../1-min/8y2g_ions_da.psf')
psf.setBox(xlength,ylength,zlength)
cor = CharmmCrdFile('../1-min/8y2g_ions_da_minimized.cor')
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


# === Parameters ===
tstep = 50000000

# === System setup ===
system = psf.createSystem(params, nonbondedMethod=PME,
                          nonbondedCutoff=1.2*nanometers,
                          constraints=HBonds,
                          rigidWater=True, ewaldErrorTolerance=0.0005)


# Load your reference PDB and system topology as you already do
ref_pdb = app.PDBFile('8y2d_ref.pdb')
system_top = psf.topology



# --- Load the reference structure with MDTraj ---
ref_traj = md.load('8y2d_ref.pdb')

# Use your system PDB or coordinates file compatible with MDTraj for topology and xyz
system_traj = md.load('8y2g_ions_da.pdb')  # Replace with your system PDB or coordinates file

def get_ca_indices(traj):
    return [atom.index for atom in traj.topology.atoms if atom.name == 'CA']

# Map residues by (chain index, resSeq) -> atom index
def get_ca_map(traj, ca_indices):
    ca_map = {}
    for i in ca_indices:
        atom = traj.topology.atom(i)
        res = atom.residue
        ca_map[(res.chain.index, res.resSeq)] = i
    return ca_map

# Build CA indices
ref_ca_indices = get_ca_indices(ref_traj)
system_ca_indices = get_ca_indices(system_traj)

ref_map = get_ca_map(ref_traj, ref_ca_indices)
system_map = get_ca_map(system_traj, system_ca_indices)

# Match residues present in both
common_keys = sorted(set(ref_map.keys()) & set(system_map.keys()))

# Extract matched indices
matched_ref_indices = [ref_map[k] for k in common_keys]
matched_system_indices = [system_map[k] for k in common_keys]

# Align structures to remove translational/rotational differences
aligned_ref = ref_traj.superpose(system_traj, atom_indices=matched_ref_indices, ref_atom_indices=matched_system_indices)
ref_ca_positions = aligned_ref.xyz[0, matched_ref_indices] * nanometer


# Total atoms in the system
n_atoms = system.getNumParticles()

# Create a full-size array of reference positions (set unused to 0,0,0)
full_ref_positions = np.zeros((n_atoms, 3)) * nanometer
for i, sys_idx in enumerate(matched_system_indices):
    full_ref_positions[sys_idx] = ref_ca_positions[i]

# Create RMSDForce using the full array and the matched system indices
rmsd_force = RMSDForce(full_ref_positions, matched_system_indices)
rmsd_force.setForceGroup(1)

# Set up CustomCVForce using only selected atoms for RMSD
custom_rmsd_force = CustomCVForce("0.5 * k * rmsd^2")
custom_rmsd_force.addCollectiveVariable("rmsd", rmsd_force)
custom_rmsd_force.addGlobalParameter("k", 500.0)

# Add the restraint force to the system
system.addForce(custom_rmsd_force)


nonbonded = None
for force in system.getForces():
    if isinstance(force, NonbondedForce):
        nonbonded = force
        break

if nonbonded is None:
    raise Exception("No NonbondedForce found in the system")

charge_scale = 0.7

# Loop through all atoms and modify charges if in specified ion residues
for i, atom in enumerate(psf.topology.atoms()):
    if atom.residue.name in ('SOD',  'CLA'):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        scaled_charge = charge * charge_scale
        nonbonded.setParticleParameters(i, scaled_charge, sigma, epsilon)
        print(f"Scaled charge of atom {i} ({atom.name} in {atom.residue.name}) from {charge} to {scaled_charge}")


# Make sure changes are applied to context later
#nonbonded.updateParametersInContext(simulation.context)

# Set up integrator and platform
integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
integrator.setConstraintTolerance(0.00001)
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'single'}

# Create simulation
simulation = Simulation(psf.topology, system, integrator, platform, properties)

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

