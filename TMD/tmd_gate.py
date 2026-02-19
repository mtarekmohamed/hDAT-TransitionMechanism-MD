from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmplumed import PlumedForce
import numpy as np
import sys


xlength = 90.2632491/10
ylength = 90.2632491/10
zlength = 118.618231/10

psf = CharmmPsfFile('../../1-min/8y2d_ions_da.psf')
#psf = PDBFile('last_frame_100ns.pdb')
psf.setBox(xlength,ylength,zlength)
cor = CharmmCrdFile('../../1-min/8y2d_ions_da_minimized.crd')

# === Load CHARMM parameters ===
params = CharmmParameterSet(
    '../../toppar/top_all36_prot.rtf',
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

# === Create system ===
system = psf.createSystem(params, nonbondedMethod=PME,
                          nonbondedCutoff=1.2*nanometers,
                          constraints=HBonds,
                          rigidWater=True, ewaldErrorTolerance=0.0005)

# === Scale ion charges ===
nonbonded = None
for force in system.getForces():
    if isinstance(force, NonbondedForce):
        nonbonded = force
        break

if nonbonded is None:
    raise Exception("No NonbondedForce found in the system")

charge_scale = 0.7
for i, atom in enumerate(psf.topology.atoms()):
    if atom.residue.name in ('SOD', 'CLA'):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        scaled_charge = charge * charge_scale
        nonbonded.setParticleParameters(i, scaled_charge, sigma, epsilon)
        print(f"Scaled charge of atom {i} ({atom.name} in {atom.residue.name}) from {charge} to {scaled_charge}")

# === Add PLUMED restraint ===
with open("plumed_ic.dat") as f:
    plumed_force = PlumedForce(f.read())
system.addForce(plumed_force)

# === Integrator and simulation setup ===
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 2.0*femtoseconds)
integrator.setConstraintTolerance(0.00001)
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'single'}
simulation = Simulation(psf.topology, system, integrator, platform, properties)
# Load checkpoint if exists (adjust path and args as needed)
checkpoint_path = f'../../2-npt/{sys.argv[4]}/{sys.argv[2]}-{sys.argv[4]}.chk'
try:
    simulation.loadCheckpoint(checkpoint_path)
except Exception as e:
    print(f"Warning: Could not load checkpoint '{checkpoint_path}': {e}")

#simulation.context.setPositions(cor.positions)
nonbonded.updateParametersInContext(simulation.context)

# === Minimize and run ===
tstep = 5000000
#simulation.loadCheckpoint("stringmethod_updated.chk")
simulation.minimizeEnergy()
simulation.reporters.append(DCDReporter("stringmethod_ic_updated.dcd", 5000))
simulation.reporters.append(StateDataReporter("stringmethod_ic_updated.log", 5000, step=True, potentialEnergy=True,
                                              temperature=True, progress=True, remainingTime=True, speed=True,
                                              totalSteps=tstep, separator='\t'))
simulation.reporters.append(CheckpointReporter("stringmethod_ic_updated.chk", 5000))

print("Running string method dynamics...")
simulation.step(tstep)

