#!/usr/bin/env python
import sys
import os
import numpy as np
from mpi4py import MPI
import synergia
import synergia.simulation as SIM
ET = synergia.lattice.element_type
MT = synergia.lattice.marker_type
PCONST = synergia.foundation.pconstants

#####################################

# Run parameters
KE = 0.00250 # kinetic energy 2.5 MeV
radius = 0.025 # aperture radius 2.5 cm
beam_current = 0.1 # [mA]
emitx = 4.3e-6 # unnormalized x RMS emittance m-rad
emity = 3.0e-6 # unnormalized y RMS emittance m-rad
std_dpop = 2.1e-3 # RMS dp/p spread
dpop_cutoff = 2.5 #sigma cutoff on generating dp/p distribution
transverse_cutoff = 3.0 # sigma cutoff on transverse distribution
harmonic = 4 # harmonic number
macroparticles = 1000
turns = 10

#######################################################

DEBUG=False

#######################################################

def print_statistics(bunch, fout=sys.stdout):

    parts = bunch.get_particles_numpy()
    print(parts.shape,  ", ", parts.size , file=fout)
    print("shape: {0}, {1}".format(parts.shape[0], parts.shape[1]))

    mean = synergia.bunch.Core_diagnostics.calculate_mean(bunch)
    std = synergia.bunch.Core_diagnostics.calculate_std(bunch, mean)
    print("mean = {}".format(mean), file=fout)
    print("std = {}".format(std), file=fout)

#######################################################
#######################################################

def save_json_lattice(lattice, jlfile='iota_lattice.json'):
    # save the lattice
    if MPI.COMM_WORLD.rank == 0:
        f = open(jlfile, 'w')
        print(lattice.as_json(), file=f)
        f.close()

################################################################################

def get_lattice():
    # read the lattice in from a MadX sequence file
    lattice = synergia.lattice.MadX_reader().get_lattice("iota", "machine.seq")

    # The sequence doesn't have a reference particle so define it here
    mass = PCONST.mp

    etot = KE + mass
    refpart = synergia.foundation.Reference_particle(1, mass, etot)

    lattice.set_reference_particle(refpart)
    # Change the tune of one plane to break the coupling resonance
    # for elem in lattice.get_elements():
    #     if elem.get_type() == ET.quadrupole:
    #         k1 = elem.get_double_attribute('k1')
    #         elem.set_double_attribute('k1', k1*0.99)
    #         break

    return lattice

################################################################################

# set a circular aperture on all elements

def set_apertures(lattice):

    for elem in lattice.get_elements():
        elem.set_string_attribute("aperture_type", "circular")
        elem.set_double_attribute("circular_aperture_radius", radius)

    return lattice

################################################################################


def get_iota_twiss(lattice):
    synergia.simulation.Lattice_simulator.CourantSnyderLatticeFunctions(lattice)
    synergia.simulation.Lattice_simulator.calc_dispersions(lattice)
    elements = lattice.get_elements()
    return elements[-1].lf

################################################################################

def create_bunch_simulator(refpart, num_particles, real_particles):
    commxx = synergia.utils.Commxx()
    sim = synergia.simulation.Bunch_simulator.create_single_bunch_simulator(
        refpart, num_particles, real_particles, commxx, 4)

    return sim

################################################################################

# determine the bunch charge from beam current
def beam_current_to_numpart(current, length, beta, harmonic):
    rev_time = length/(beta*PCONST.c)
    total_charge = current*rev_time/PCONST.e
    # charge is divided into bunches by harmonoic number
    return total_charge/harmonic
    

################################################################################

# populate a matched distribution using emittances, dispersions and the lattice
def populate_matched_distribution(lattice, bunch, emitx, betax, Dx, emity, betay, stddpop):
    # populate entire bunch with 0
    localnum = bunch.get_local_num()
    lp = bunch.get_particles_numpy()
    lp[0:localnum, 0:6] = 0.0
    bunch.checkin_particles()

    return

    # the remainder of this routine is disabled

    # # construct standard deviations
    # stdx = np.sqrt(emitx*betax + stddpop**2*Dx**2)
    # stdy = np.sqrt(emity*betay)
    # beta = lattice.get_reference_particle().get_beta()
    # print('stdx: ', stdx)
    # print('stdy: ', stdy)
    # print('stdcdt: ', stddpop)

    # map = synergia.simulation.Lattice_simulator.get_linear_one_turn_map(lattice)
    # corr_matrix =  synergia.bunch.get_correlation_matrix(map,
    #                                                           stdx,
    #                                                           stdy,
    #                                                           stddpop,
    #                                                           beta,
    #                                                           (0, 2, 5))

    # dist = synergia.foundation.PCG_random_distribution(1234567, synergia.utils.Commxx.World.rank())

    # means = np.zeros(6)
    # tc = transverse_cutoff
    # lc = dpop_cutoff
    # limits = np.array([tc, tc, tc, tc, lc, lc])
    # synergia.bunch.populate_6d_truncated(dist, bunch, means,
    #                                      corr_matrix, limits)

################################################################################


def register_diagnostics(sim):
    # diagnostics
    diag_full2 = synergia.bunch.Diagnostics_full2("diag.h5")
    sim.reg_diag_per_step(diag_full2)

    diag_bt = synergia.bunch.Diagnostics_bulk_track("tracks.h5", 100, 0)
    sim.reg_diag_per_step(diag_bt)

################################################################################

def get_propagator(lattice):
    if  DEBUG:
        print('get_propagator operating on lattice: ', id(lattice))

    steps = 1

    stepper = synergia.simulation.Independent_stepper_elements(1)

    propagator = synergia.simulation.Propagator(lattice, stepper)

    if DEBUG:
        print('lattice from propagator: ', id(propagator.get_lattice()))

    return propagator

################################################################################


################################################################################

def main():

    logger = synergia.utils.Logger(0)

    lattice = get_lattice()
    print('Read lattice, length = {}, {} elements'.format(lattice.get_length(), len(lattice.get_elements())), file=logger)
    lattice_length = lattice.get_length()

    print('RF cavity frequency should be: ', 4*lattice.get_reference_particle().get_beta() * PCONST.c/lattice_length)

    # set the aperture early
    set_apertures(lattice)

    state = SIM.Lattice_simulator.tune_circular_lattice(lattice)
    print('state: ', state)
    print('length maybe: ', state[4]*lattice.get_reference_particle().get_beta())

    for elem in lattice.get_elements():
        if elem.get_type() == ET.rfcavity:
            print('RF cavity: ', elem, file=logger)

    refpart = lattice.get_reference_particle()

    energy = refpart.get_total_energy()
    momentum = refpart.get_momentum()
    gamma = refpart.get_gamma()
    beta = refpart.get_beta()

    print("energy: ", energy, file=logger)
    print("momentum: ", momentum, file=logger)
    print("gamma: ", gamma, file=logger)
    print("beta: ", beta, file=logger)

    iota_twiss = get_iota_twiss(lattice)
    lf = iota_twiss
    print('IOTA lattice Twiss parameters:', file=logger)
    print(f'beta x: {lf.beta.hor}, alpha x: {lf.alpha.hor}, x tune: {lf.psi.hor/(2*np.pi)}', file=logger)
    print(f'disp x: {lf.dispersion.hor}, dprime x: {lf.dPrime.hor}', file=logger)
    print(f'beta y: {lf.beta.ver}, alpha y: {lf.alpha.ver}, y tune: {lf.psi.ver/(2*np.pi)}', file=logger)
    print(f'disp y: {lf.dispersion.ver}, dprime y: {lf.dPrime.ver}', file=logger)
    
    map = SIM.Lattice_simulator.get_linear_one_turn_map(lattice)
    l, v = np.linalg.eig(map)
    print("eigenvalues: ", file=logger)
    for z in l:
        print("|z|: ", abs(z), " z: ", z, " tune: ", np.log(z).imag/(2.0*np.pi), file=logger)

    bunch_charge = beam_current_to_numpart(beam_current, lattice_length, beta, harmonic)
    bunch_sim = create_bunch_simulator(refpart, macroparticles, bunch_charge)

    print('beam current: ', beam_current, ' mA', file=logger)
    print('bunch created with ', macroparticles, ' macroparticles', file=logger)
    print('bunch charge: ', bunch_charge, file=logger)

    #### generate bunch
    populate_matched_distribution(lattice, bunch_sim.get_bunch(0,0),
                                  emitx, lf.beta.hor, lf.dispersion.hor,
                                  emity, lf.beta.ver,
                                  std_dpop)

    print_statistics(bunch_sim.get_bunch(0, 0), logger)


    register_diagnostics(bunch_sim)


    ####  stepper and collective operators

    propagator = get_propagator(lattice)

    # logger for simulation
    simlog = synergia.utils.parallel_utils.Logger(0, 
            synergia.utils.parallel_utils.LoggerV.INFO_TURN)
            #synergia.utils.parallel_utils.LoggerV.INFO)
            #synergia.utils.parallel_utils.LoggerV.INFO_STEP)


    propagator.propagate(bunch_sim, simlog, turns)



if __name__ == "__main__":

    main()
