# a vacuum simulation of alanine dipeptide
from openmm import *
from openmm.app import *
from openmm.unit import *
import mdtraj
from sys import stdout
import simtk.unit as unit
import uuid 
import numpy as np 
import os

# get pdb 
# set up force reporter 
class ForceReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(unit.kilojoules/unit.mole/unit.nanometer)
        for f in forces:
            self._out.write('%g %g %g ' % (f[0], f[1], f[2]))
        self._out.write('\n')
        self._out.flush()

def get_simulation_params(config): 
    pdb = PDBFile("alanine-dipeptide.pdb")
    print(pdb.topology)

    # add accelerator 
    platform = Platform.getPlatformByName(config['platform'])
    properties = {'DeviceIndex': '0'}

    # set up system and do energy minimization
    forcefield = ForceField(config['forcefield']) # no water! 
    system = forcefield.createSystem(pdb.topology, nonbondedCutoff=3 * nanometer, constraints=HBonds)
    integrator = LangevinIntegrator(config['temperature'] * kelvin, \
                                    config['friction'] / picosecond, \
                                    config['timestep'] * picoseconds)
    return pdb, system, integrator, platform, properties

def assemble_simulation(pdb, system, integrator, platform, properties):
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    # minimize energy
    simulation.minimizeEnergy()
    return simulation

def get_metadynamics(system, config, output_dir): 
    # add forces 
        phi_idx, psi_idx = config['metadynamics']['cv_1']['atom_indices'], config['metadynamics']['cv_2']['atom_indices']
        phi, psi = CustomTorsionForce("theta"), CustomTorsionForce("theta")
        psi.addTorsion(*psi_idx)
        phi.addTorsion(*phi_idx)
        bias_phi = metadynamics.BiasVariable(force=phi,
                                             minValue=-np.pi, maxValue=np.pi,
                                             biasWidth=config['metadynamics']['cv_1']['biasWidth'],
                                             periodic=config['metadynamics']['cv_1']['periodic'],
                                             gridWidth=config['metadynamics']['cv_1']['gridWidth'])
        bias_psi = metadynamics.BiasVariable(force=psi, \
                                             minValue=-np.pi, maxValue=np.pi,
                                             biasWidth=config['metadynamics']['cv_2']['biasWidth'], 
                                             periodic=config['metadynamics']['cv_2']['periodic'],
                                             gridWidth=config['metadynamics']['cv_2']['gridWidth'])
        meta = Metadynamics(system, 
                                    variables=[bias_phi, bias_psi], 
                                    temperature=config['temperature']*unit.kelvin, 
                                    biasFactor=config['metadynamics']['biasFactor'], 
                                    height=config['metadynamics']['height'], 
                                    frequency=config['metadynamics']['frequency'], 
                                    saveFrequency=config['metadynamics']['frequency'], 
                                    biasDir=output_dir)
        return meta

def append_reporters(simulation, report_dict):
    simulation.reporters.append(DCDReporter(report_dict['traj_path'], report_dict['frequency']))
    simulation.reporters.append(
        StateDataReporter(stdout, 10*report_dict['frequency'], step=True, potentialEnergy=True, temperature=True, elapsedTime=True)
    )
    simulation.reporters.append(
        StateDataReporter(
            report_dict['potentials_path'], report_dict['frequency'], step=True, potentialEnergy=True)
    )
    simulation.reporters.append(
        ForceReporter(report_dict['forces_path'], report_dict['frequency'])
    )
    return simulation

def get_savefiles(): 
    # generate hex key, create directory for output 
    hexkey = uuid.uuid4().hex
    # make directory: 
    output_dir = f"outputs/alanine_vacuum_{hexkey}"
    os.makedirs(output_dir, exist_ok=True) 
    
    # now set up reporters--report positions, potential, and forces. 
    traj_path = os.path.join(output_dir, "traj_alanine_vacuum.dcd")
    potentials_path = os.path.join(output_dir, "traj_alanine_vacuum_potentials.txt")
    forces_path =  os.path.join(output_dir, "traj_alanine_vacuum_forces.txt")
    output_config_path = os.path.join(output_dir, "simulation_config.yaml")
    return output_dir, traj_path, potentials_path, forces_path, output_config_path

def writeconfig(config, output_config_path):
    import yaml
    with open(output_config_path, 'w') as file:
        yaml.dump(config, file)

def simulate(config): 
    # get save files
    output_dir, traj_path, potentials_path, forces_path, output_config_path = get_savefiles()
    print("Output directory:", output_dir)
    # set up simulation
    pdb, system, integrator, platform, properties = get_simulation_params(config)
    breakpoint()
    # add metadynamics if specified
    if config['metadynamics']['simulate']: 
        meta = get_metadynamics(system, config, output_dir)
    simulation = assemble_simulation(pdb, system, integrator, platform, properties)
    breakpoint()
    # append reporters 
    frequency = config['frequency'] 
    report_dic = {'traj_path': traj_path, 
                   'potentials_path': potentials_path, 
                   'forces_path': forces_path, 
                   'frequency': frequency}
    simulation = append_reporters(simulation, report_dic)
    print("Set up reporters, now starting simulation!")
    # run simulation
    breakpoint()
    nsteps = config['nsteps']  # 1 ns
    if config['metadynamics']['simulate']: 
        meta.step(simulation, nsteps)
        # save free energy: 
        np.savez(os.path.join(output_dir, "free_energy.npz"), free_energy=meta.getFreeEnergy())
    else:
        simulation.step(nsteps)
    print("Done with simulation, now writing config!")
    # write config to output dir
    writeconfig(config, output_config_path)
    print("Done writing config!")

if __name__ == "__main__":
    import yaml
    with open("simulation_config_metadynamics.yaml", 'r') as file:
        config = yaml.safe_load(file)
    simulate(config)