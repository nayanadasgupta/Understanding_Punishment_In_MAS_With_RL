# Understanding Punishment In Multi-Agent Systems Using Reinforcement Learning

Code to replicate results of Final Year Project for MEng Computer Science.

## Set Up

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements from the `requirements.txt` file.

```bash
source ./venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```
Every Simulation folder e.g. `Simulations/DP_S/Centipede` should contain a `Logs` and a `Results` folder before running the simulation.

## Simulations

Run simulations from the root folder as so:

```bash
python Simulations/TPP_S/Centipede/MultipleRunner_TPPSelect_playPunishRep_repInPlayState_centipede.py
```

## Analysis

Use the metric and plotting functions in the `analysis.py` file to analyse the results.


## Naming Conventions

`play_MECHANISM_REPINFO_REPSTATE_CONTEXT` denotes a single simulation of the social mechanism `MECHANISM` when reputation is calculated using the information defined in`REPINFO` and reputational information is added to `REPSTATE`. This simulation is performed with the following `CONTEXT` e.g. social dilemma.

`MultipleRunner_MECHANISM_REPINFO_REPSTATE_CONTEXT` runs a single simulation multiple times to allow for rolling mean and a 95\% confidence interval to be calculated.  