# EpiCommute

Simulate an epidemic on a metapopulation network with commuter-type mobility mobility.

The model is used and defined in the following publication:

> "COVID-19 lockdown induces structural changes in mobility networks -- Implication for mitigating disease dynamics"
> Frank Schlosser, Benjamin F. Maier, David Hinrichs, Adrian Zachariae, Dirk Brockmann
> [https://arxiv.org/abs/2007.01583](https://arxiv.org/abs/2007.01583)

## Install

```python
python setup.py install
```

## Usage example

```python
>>> model = SIRModel(mobility, subpopulation_sizes)
>>> results = model.run_simulation(VERBOSE=True)
Starting Simulation ...
Simulation completed
Time: 0min 3.35s
```

More examples are given in the notebooks at `/examples`.

## Model description

The code simulates an SIR epidemic on a subpopulation network,
where subpopulations are connected by commuter-type mobility.

(A detailed descriptions of the model is given in the mauscript linked above).

### Mobility

Movement of individuals between subpopulation is implemented using
commuter-type dynamics. This means that each individual lives at a home location
i, and works at a work location j.

How the individuals are distributed among the compartments is determined
by an origin-destination mobility matrix F of size M x M, which contains
the number of individuals commuting between pairs of locations.

The population in the system is then distributed into M x M compartments,
where compartment ij are those individuals living at i and working at j.

### Infection dynamics

Epidemic spread is simulated using the SIR model, consisting of susceptibles S,
infecteds I and recovereds R.

The infection step is subdivided in two phases of equal length:

- In the `home` phase, each individual has a chance to get infected at their home location i.
- In the `work` phase, infections can take place at the work locations.

### Quarantine/lockdown effects

The model can consider changes in absolute mobility flux (for example
due to lockdown effects). For this, it is a assumed that a matrix
`mobility` is provided with the current (possibly reduced) number
of commuters, and a matrix `mobility_baseline` with the number of commuters during normal times.

Changes in mobility flux are taken into account in two different scenarios:

- In the `isolation` scenario, it is assumed that a reduction in mobility
  means that individuals are effectively removed from the system
- In the `distancing` scenario, a reduction in mobility instead leads
  to a reduction in the effective transmission rate in the system.

A more detailed description of the scenarios and the model can be found in the publication.
