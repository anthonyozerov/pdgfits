# Code to do the averages and fits done by the PDG

This codebase is designed to perform the averages and fits done by the [Particle Data Group](https://pdg.lbl.gov/). Note that it pulls directly from the internal PDG database, so unless you have access to it, you won't be able to run the code. Note also that the implemented methods are what I (Anthony) think works well, and not what the PDG currently does.

## Structure

The code unifies *averages* and *fits* in a likelihood-based framework. Both averages and fits are done by obtaining an MLE; equivalently, minimizing the $\chi^2$ function. The core average logic is in `avg.py`, and the core fit logic is in `fit.py`. Beyond that:

- `query.py` contains functions that query the PDG database.
- `preprocess.py` preprocesses the data from the PDG database.
- `parser.py` contains string processing to parse measurements and their units.
- `build_funcs.py` build functions that map parameters to measurands.
- `build_chi2.py` builds the $\chi^2$ function that maps parameters to a $\chi^2$ value (for a certain set of measurements).

## Usage

To run all of the averages, run `run_avgs.py`. To run all of the fits, run `run_fits.py`.
