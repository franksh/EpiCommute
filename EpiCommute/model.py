"""
Provides classes for the epidemiological model
"""
import numpy as np
from scipy.stats import binom, expon


class SIRModel():
    """
    An SIR metapopulation model with commuter-type mobility.

    There are M subpopulations, and M commuter compartments within each subpopulation.

    Two different modes:

    Parameters
    ----------
    mobility: numpy.ndarray
        The origin-destination mobility matrix of the system.
        Has to be quadratic, i.e. of shape MxM.

    subpopulation_sizes: numpy.ndarray 
        The population sizes in the districts M.
        Shape 1xM.

    mobility_baseline: numpy.ndarray
        An optional baseline mobiliy matrix.

    quarantine_mode: str
        Which type of quarantine to use.
        Possibilities: 'isolation', 'distancing'

    outbreak_source: str or int or None
        Where to seed the infection.
        Possible values:
         - 'random': A random subpopulation is chosen.
         - int: If an integer m is given, the corresponding subpopulation
            m is chosen as the infection site.
         - None: Defaults to random subpopulation.

    Attributes
    ----------
    population: numpy.ndarray
        The population of the system

    kappa: numpy.ndarray
        The quarantine factor


    Example
    -------
    .. code:: python
        
        >>> sir = SIRModel(mobility, population)
        >>> sir.run_simulation()
        [ "S", "I", "R" ]
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


    def _initialize(self):
        """
        Initialize the model
        """
        self._check_if_input_data_valid()

        self._initialize_population()

        if self.quarantine_mode:
            self._calculate_quarantine_factor()


    def _check_if_input_data_valid(self):
        """
        Check if the given input data is valid
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
                    "Outbreak source has to be an integer, 'random', or None"
            if type(self.outbreak_source) is int:
                assert self.outbreak_source < self.mobility.shape[0],\
                    "Outbreak source has to be within system size"

    def _initialize_population(self):
        """
        Create the commuter population.

        The population, where the sizes in each subpopulation is given,
        is distributed into the commuting subpopulations according to
        the mobility.
        """

        # Normalize the mobility matrix for each row (origin)
        mobility_subpops = self.mobility.sum(axis=1)
        mobility_normalized = (self.mobility.T / mobility_subpops).T

        # Create the population-commuter table
        population = (mobility_normalized.T * self.subpopulation_sizes).T
        self.population = np.round(population).astype(int)


    def _calculate_quarantine_factor(self):
        """ 
        Calculates the influence of quarantine.

        Only applied if a "quarantine_mode" is specified. Then, kappa is
        a dynamic factor that accounts for the influence of quarantine,
        and is calculated here.

        """
        mob_baseline = self.mobility_baseline.copy()
        mob_baseline[mob_baseline == 0] = 1

        # Calculate kappa
        self.kappa = self.mobility / mob_baseline



    # SIMULATION

    def reset_initialize_simulation(self):
        """ 
        Resets the simulation and initializes
        """
        # If isolation: Move a fraction *kappa* of S to R compartment initially
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

  

    def _seed_infection(self):
        """
        Initialize the infection seed.

        Choses a subpopulation where the infection starts, and distributes
        the initial number of infected I0 among the commuter compartments.
        """
        M = self.mobility.shape[0] # Number of compartments

        # Determine infected subpopulations
        if self.outbreak_source in ['random', None]:    
            idx = np.random.choice(np.arange(M))
        else:
            idx = self.outbreak_source
        
        infected_subpopulation_size = np.sum(self.S[idx])
        if infected_subpopulation_size < self.I0:
            raise ValueError(f"Cannot seed infection: Subpopulation {idx} "+\
                f"contains only {infected_subpopulation_size} individiuals, "+\
                f"but I0={self.I0}")

        # Choose the commuter compartments within the location idx
        # infected_compartments = np.random.choice(M,
        #                         p=self.S[idx]/self.S[idx].sum(),
        #                         replace=True, size=self.I0)
        # # Assign the infecteds
        # for i in infected_compartments:
        #     self.S[idx][i] -= 1
        #     self.I[idx][i] += 1

        for infected in range(self.I0):
            infected_compartment = np.random.choice(M, 
                                            p=self.S[idx]/self.S[idx].sum())
            # Assign the infected
            self.S[idx][infected_compartment] -= 1
            self.I[idx][infected_compartment] += 1
                                

    def run_simulation(self):
        """ 
        Run the simulation
        """
        self.reset_initialize_simulation()

        if self.VERBOSE:
            print("Starting Simulation ...")
            import time
            time_start = time.time()

        self._seed_infection()

        t = 0
        while t < self.T_max + self.dt:
            
            self._update_infection()
            
            # Save observables
            remainder = t % self.dt_save
            is_save_time = np.allclose(remainder, 0.0, atol=1e-4) or np.allclose(remainder, self.dt_save, atol=1e-4)
            if is_save_time:
                self._save_observables(t)

            # Update time
            t += self.dt


        # self._save_observables(t=0)
        # for t in self.tqdm_counter(np.arange(1, self.T_max)):
        #     # Infection
        #     self._update_infection()

        #     if np.allclose(t % self.dt_save, 0.0):
        #         self._save_observables(t)

        if self.VERBOSE:
            print("Simulation completed")
            minutes, seconds = divmod(time.time() - time_start, 60)
            print("Time: {:.0f}min {:.2f}s".format(minutes, seconds))
        return self.observables

    def _update_infection(self):
        """
        Update the infection dynamics
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
        lambda_home = self.beta * I_ij_sumj / N_ij_sumj
        # Work force of infection
        I_ji_sumj = I.sum(axis=0)
        N_ji_sumj = S.sum(axis=0) + I.sum(axis=0) + R.sum(axis=0)
        lambda_work = self.beta * I_ji_sumj / N_ji_sumj

        M = self.S.shape[0] # number of subpopuations
        for i in range(M):
            # Normal infection rate
            if self.quarantine_mode in [None, 'isolation']:
                lambda_home_eff = lambda_home[i]
                lambda_work_eff = lambda_work[i]
            # Distancing scenario: Modify transmission rate linearly with kappa
            elif self.quarantine_mode == 'distancing':
                lambda_home_eff = self.kappa[i, :] * lambda_home[i]
                lambda_work_eff = self.kappa[:, i] * lambda_work[i]
            # Calculate infections
            # Home force of infection
            try:
                dSI_i = binom.rvs(S[i], expon.cdf(lambda_home_eff * self.dt))
                S[i] -= dSI_i
                I[i] += dSI_i
                # Work force of infection
                dSI_i = binom.rvs(S[i], expon.cdf(lambda_work_eff * self.dt))
                S[i] -= dSI_i
                I[i] += dSI_i
            except ValueError as e:
                print(e)
                breakpoint()

        # Calculate recoveries
        dIR = binom.rvs(I, expon.cdf(self.mu * self.dt))
        I = I - dIR
        R = R + dIR

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
            # The arrival time of the epidemic in each subpopulation

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
    subpopulation_sizes = np.random.randint(1,100,M)

    model = SIRModel(mobility, subpopulation_sizes)
    # breakpoint()

    # Dummy with baseline
    # Create dummy data
    M = 10 # Number of locations
    mobility = np.random.rand(M, M)
    subpopulation_sizes = np.random.randint(1,100,M)

    quarantine_mode = 'isolation'
    # mobility_baseline = mobility * 2
    mobility_baseline = mobility * np.random.rand(M,M)

    model = SIRModel(mobility, subpopulation_sizes, mobility_baseline, 
                        quarantine_mode=quarantine_mode)


