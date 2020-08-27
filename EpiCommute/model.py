"""
Provides the class for the epidemiological model
"""
import numpy as np
from scipy.stats import binom, expon

class SIRModel():
    """
    An SIR metapopulation model with commuter-type mobility.

    This model used in the following paper (a detailed description of the model
    can be found there):

    'COVID-19 lockdown induces structural changes in mobility
    networks -- Implication for mitigating disease dynamics'
    Frank Schlosser, Benjamin F. Maier, David Hinrichs,
    Adrian Zachariae, Dirk Brockmann
    https://arxiv.org/abs/2007.01583

    See also notebooks in /examples for usage examples.

    The system is composed of M subpopulations. Individuals can commute 
    between pairs of subpopulations. The weight of commuter flows is given
    by the mobility-matrix of shape M x M.

    The model can consider changes in absolute mobility flux (for example
    due to lockdown effects). For this, it is a assumed that the matrix
    'mobility' contains the current, changed flux, and the matrix
    'mobility_baseline' contains the flow during normal times.

    Changes in mobility flux are taken into account in two different scenarios:
     - In the 'isolation' scenario, it is assumed that a reduction in mobility
        means that individuals are effectively removed from the system
     - In the 'distancing' scenario, a reduction in mobility instead leads
        to a reduction in the effective transmission rate in the system.
    A more detailed description of the scenarios and the model can be found
    in the publication.

    Parameters
    -----------
    mobility: numpy.ndarray
        The origin-destination mobility matrix of the system. Has to symmetric,
        of shape MxM. Each entry ij depicts the strength of the commuter flow
        between subpopulations i and j. Self-flows ii should be included,
        indicating people that live and commute in the same subpopulation.

    subpopulation_sizes: numpy.ndarray 
        The population sizes in the subpopulations, of shape 1xM.

    mobility_baseline: None or numpy.ndarray (default None)
        An optional baseline mobility matrix. If given, it is assumed that
        the current mobility is a change in flux from the baseline matrix.
        The change is implemented using one of the quarantine scenarios.

    quarantine_mode: None or str (default None)
        Which type of quarantine scenario to use, see description above.
        Valid options:: 'isolation', 'distancing'.

    outbreak_source: str or int or None (default None)
        Where to seed the infection.
        Possible values:
         - 'random': A random subpopulation m is chosen.
         - int: If an integer m is given, the corresponding subpopulation
            m is chosen as the infection site.
         - None: Defaults to random subpopulation.

    T_max: int (default 100)
        The time until which to run the simulation.

    dt: float (default 0.1)
        The simulation time increment. Lower values reduce stochastic noise.

    dt_save: float (default 1)
        The simulation time interval at which to save observables.
    
    mu: float (default 1/8)
        The recovery rate in the SIR model.

    R0: float (default 3.0)
        The basic reproduction number.

    I0: int (default 10)
        The initial number of infected.

    save_observables: list of str
        Which observables to save. Possible options:
         - 'epi_total': The total number of S, I and R in the system at each time.
         - 'epi_subpopulations': The number of S, I and R in each subpopulation.
         - 'arrival_times': A list of arrival times, i.e. when the epidemic
                arrived in each of the subpopulations.

    VERBOSE: bool (default False)
        Whether to print information on the running simulation.

    Attributes
    ----------
    population: numpy.ndarray
        The population of the system, divided into the M subpopulations and
        commuter compartments. Matrix of shape MxM, where the entry ij are the
        individual in subpopulation i which commute to j.

        The population matrix is created by normalizing the mobility matrix
        to the number of individuals in each subpopulation (given by the 
        vector subpopulation_sizes).

    kappa: numpy.ndarray
        The quarantine factor, calculated as the ratio of mobility and
        mobility_baseline. It is used in the simulation in different ways
        depending on which quarantine scenario is chosen.

    observables: dict
        A dictionary of observables which are returned after the simulation
        is run.


    Example
    -------
    .. code:: python
        
        >>> model = SIRModel(mobility, subpopulation_sizes)
        >>> results = model.run_simulation(VERBOSE=True)
        Starting Simulation ...
        Simulation completed
        Time: 0min 3.35s
    """
    def __init__(self,
                mobility,
                subpopulation_sizes,
                mobility_baseline = None,
                quarantine_mode = None,
                outbreak_source='random',
                T_max=100,
                dt=0.1,
                dt_save=1,
                mu=1/8,
                R0=3.0,
                I0=10,
                save_observables=[
                    'epi_subpopulations',
                    'epi_total',
                    'arrival_times'],
                VERBOSE=False
                ):

        # Mobility data
        self.mobility = mobility
        self.subpopulation_sizes = subpopulation_sizes
        self.mobility_baseline = mobility_baseline
        self.quarantine_mode = quarantine_mode

        # Simulation variables
        self.population = None
        self.kappa = np.ones(mobility.shape)
        self.observables = {}

        # Simulation parameters
        self.outbreak_source = outbreak_source
        self.T_max = T_max
        self.dt = dt
        self.dt_save = dt_save
        self.save_observables = save_observables

        # Epidemiological parameters
        self.mu = mu            # recovery rate
        self.R0 = R0            # reproduction number
        self.beta = mu * R0     # infection rate
        self.I0 = I0            # initial number of infected

        self.VERBOSE = VERBOSE

        self._initialize()

    ## INITIALIZATION ##########################################

    def _initialize(self):
        """
        Initialize the model for given input data.
        """
        self._check_if_input_data_valid()

        self._initialize_population()

        if self.quarantine_mode:
            self._calculate_quarantine_factor()

    def _check_if_input_data_valid(self):
        """
        Some basic checks whether the input parameters and data are
        in the correct formats.
        """
        # Mobility matrix
        assert(type(self.mobility) is np.ndarray)
        assert(self.mobility.shape[0] == self.mobility.shape[1])

        # Population
        assert(len(self.subpopulation_sizes) == self.mobility.shape[0])

        if self.mobility_baseline is not None:
            assert(type(self.mobility_baseline) is np.ndarray)
            assert(self.mobility_baseline.shape == self.mobility.shape)
            assert(self.quarantine_mode is not None)

        if self.quarantine_mode:
            assert(self.quarantine_mode in ['isolation', 'distancing'])
            assert(self.mobility_baseline is not None)

        if self.outbreak_source:
            assert ((type(self.outbreak_source) is int) or
                    (self.outbreak_source=='random')),\
                    "Outbreak source has to be an integer, 'random', or None."
            if type(self.outbreak_source) is int:
                assert self.outbreak_source < self.mobility.shape[0],\
                    "Outbreak source has to be within system size."

    def _initialize_population(self):
        """
        Create the population matrix.

        Creates the MxM matrix 'population' containing the number of
        individuals in each subpopulation-commuter compartment.
        """
        # Normalize the mobility matrix for each row (origin)
        mobility_subpops = self.mobility.sum(axis=1)
        mobility_normalized = (self.mobility.T / mobility_subpops).T

        # Create the population-commuter table
        population = (mobility_normalized.T * self.subpopulation_sizes).T
        self.population = np.round(population).astype(int)

    def _calculate_quarantine_factor(self):
        """ 
        Calculates the influence factor 'kappa' of quarantine.

        Only applied if a "quarantine_mode" is specified. Then, kappa is
        a dynamic factor that accounts for the influence of quarantine,
        and is calculated here.
        """
        # Get baseline mobility
        mob_baseline = self.mobility_baseline.copy()
        mob_baseline[mob_baseline == 0] = 1

        # Calculate kappa for each subpopulation i
        n_subpops = mob_baseline.shape[0]
        for i in range(n_subpops):
            k_outgoing = np.sum(self.mobility[i, :]) / np.sum(mob_baseline[i, :])
            k_ingoing = np.sum(self.mobility[:, i]) / np.sum(mob_baseline[:, i])
            self.kappa[i, :] = 0.5*(k_outgoing + k_ingoing)

    ## SIMULATION ##########################################

    def reset_initialize_simulation(self):
        """ 
        Resets the simulation and prepare a new one.

        This initializes the compartments S, I and R (and applies
        quarantine isolation effects if appropriate), seeds the infection,
        prepares the result observables.
        """
        # If isolation scenario:
        # Move a fraction *kappa* of S to R compartment initially
        if self.quarantine_mode == 'isolation':
            self.S = np.round(self.population * self.kappa).astype(int)
            self.R = np.round(self.population * (1-self.kappa)).astype(int)
        else:
            self.S = self.population.copy()
            self.R = np.zeros(self.population.shape, dtype=int)

        self.I = np.zeros(self.population.shape, dtype=int)

        self.observables = {}
        self.observables['t'] = []
        if 'epi_subpopulations' in self.save_observables:
            self.observables['S'] = []
            self.observables['I'] = []
            self.observables['R'] = []
        if 'epi_total' in self.save_observables:
            self.observables['S_total'] = []
            self.observables['I_total'] = []
            self.observables['R_total'] = []
        if 'arrival_times' in self.save_observables:
            M = self.population.shape[0] # number of subpopulations
            self.observables['T_arrival'] = np.ones(M) * self.T_max

        self._seed_infection()

    def _seed_infection(self):
        """
        Initialize the infection seed.

        Choses a subpopulation m where the infection starts, and distributes
        the initial number of infected I0 among the commuter compartments.
        """
        M = self.mobility.shape[0]

        # Determine infected subpopulation
        if self.outbreak_source in ['random', None]:    
            idx = np.random.choice(np.arange(M))
        else:
            idx = self.outbreak_source
        
        # Check if population big enough to seed infection
        infected_subpopulation_size = np.sum(self.S[idx])
        if infected_subpopulation_size < self.I0:
            raise ValueError(f"Cannot seed infection: Subpopulation {idx} "+\
                f"contains only {infected_subpopulation_size} individiuals, "+\
                f"but I0={self.I0}")
        
        # Distribute the infected among the commuter-subcompartments
        P = self.population[idx] / self.population[idx].sum()
        for infected in range(self.I0):
            infected_compartment = np.random.choice(M, p=P)
            # Assign the infected
            self.S[idx][infected_compartment] -= 1
            self.I[idx][infected_compartment] += 1
                                

    def run_simulation(self):
        """ 
        Run the simulation.

        Simulates the SIR epidemic up until time T_max. Simulation dynamics
        are updated in increments of dt. At intervals dt_save, observables
        are saved.
        """
        self.reset_initialize_simulation()

        if self.VERBOSE:
            print("Starting Simulation ...")
            import time
            time_start = time.time()

        t = 0
        while t < self.T_max + self.dt:            
            # Save observables
            remainder = t % self.dt_save
            is_save_time = np.allclose(remainder, 0.0, atol=1e-4) or np.allclose(remainder, self.dt_save, atol=1e-4)

            if is_save_time:
                self._save_observables(t)

            # Update infection dynamics
            self._update_infection()

            # Update time
            t += self.dt

        if self.VERBOSE:
            print("Simulation completed")
            minutes, seconds = divmod(time.time() - time_start, 60)
            print("Time: {:.0f}min {:.2f}s".format(minutes, seconds))
        return self.observables

    def _update_infection(self):
        """
        Update the infection dynamics for one time increment [t, t+dt].
        """
        # Do nothing if no infected (to speed up simulation)
        if np.sum(self.I) < 1:
            return
        
        S = self.S
        I = self.I
        R = self.R

        # Home force of infection
        I_ij_sumj = I.sum(axis=1)
        N_ij_sumj = S.sum(axis=1) + I.sum(axis=1) + R.sum(axis=1)
        lambda_home = 0.5 * self.beta * I_ij_sumj / N_ij_sumj
        # Work force of infection
        I_ji_sumj = I.sum(axis=0)
        N_ji_sumj = S.sum(axis=0) + I.sum(axis=0) + R.sum(axis=0)
        lambda_work = 0.5 * self.beta * I_ji_sumj / N_ji_sumj

        M = self.S.shape[0] # number of subpopulations
        for i in range(M):
            # Normal infection rate
            if self.quarantine_mode in [None, 'isolation']:
                lambda_home_eff = lambda_home[i]
                lambda_work_eff = lambda_work
            # Distancing scenario: Modify transmission rate linearly with kappa
            elif self.quarantine_mode == 'distancing':
                lambda_home_eff = self.kappa[i, :] * lambda_home[i]
                lambda_work_eff = self.kappa[:, i] * lambda_work
            # Calculate infections
            # Home force of infection
            dSI_i = binom.rvs(S[i], expon.cdf(
                (lambda_home_eff + lambda_work_eff) * self.dt))
            # Calculate recoveries
            dIR_i = binom.rvs(I[i], expon.cdf(self.mu * self.dt))
            
            # Update system
            S[i] = S[i] - dSI_i
            I[i] = I[i] + dSI_i - dIR_i
            R[i] = R[i] + dIR_i

        self.S = S
        self.I = I
        self.R = R

    def _save_observables(self, t):
        """
        Save the observables
        """
        self.observables['t'].append(t)
        
        if 'epi_total' in self.save_observables:
            # The total number of S, I, R in the system
            total_population = np.sum(self.population)
            self.observables['S_total'].append( self.S.sum() / total_population)
            self.observables['I_total'].append( self.I.sum() / total_population)
            self.observables['R_total'].append( self.R.sum() / total_population)
        if 'epi_subpopulations' in self.save_observables:
            # The number of S, I, R in each subpopulation
            subpopulations = self.population.sum(axis=1)
            self.observables['S'].append( self.S.sum(axis=1) / subpopulations)
            self.observables['I'].append( self.I.sum(axis=1) / subpopulations)
            self.observables['R'].append( self.R.sum(axis=1) / subpopulations)
        if 'arrival_times' in self.save_observables:
            # Save the arrival time of the epidemic in each subpopulation
            #
            # The threshold for the number of infected when
            # the epidemic has reached a subpopulation
            I_threshold = 0.001
            t_arrivals_new = ((self.I.sum(axis=1) / self.population.sum(axis=1)
                              > I_threshold) * t)
            t_arrivals_new[t_arrivals_new == 0] = self.T_max
            self.observables['T_arrival'] = np.min(
                [self.observables['T_arrival'], t_arrivals_new], axis=0)

if __name__ == '__main__':
    # Create dummy data
    M = 10 # Number of locations
    mobility = np.random.rand(M, M)
    subpopulation_sizes = np.random.randint(20,100,M)

    # Create model
    model = SIRModel(mobility, subpopulation_sizes, VERBOSE=True)

    # Run simulation
    results = model.run_simulation()

    
