# dreamerv3-pytorch
A clean, open-source documented PyTorch implementation of the [Dreamer V3 algorithm](https://github.com/danijar/dreamerv3).

The goal of this project is to provide a modular and (relatively) easy to understand codebase that can easily be modified and/or built upon by other researchers.

## Installation
The easiest way to install is via [uv](https://docs.astral.sh/uv/), the best Python dependency manager (in my opinion). Note, this has only been tested on Linux (namely Ubuntu 20/22 and Fedora 43).

Simply clone and navigate to the root of the repo. Then, run:

```bash
uv venv --python 3.11
```

to create a venv for Python 3.11, the required version for this project.

If you don't have python-3.11 installed, one can easily install it via:

```bash
uv python install 3.11
```

Then, run the command:

```bash
uv sync
```

which will install all of the dependencies as given in the `uv.lock` file. If you are missing any system dependencies, install as prompted.

## Usage
Configuration is managed via [Hydra](https://hydra.cc/docs/intro/), whose entry point is the `main.py` file. To run the default model parameters on the default deep-mind-control-suite task, with visual inputs, simply run:

```bash
python main.py env=dmc
```

Logging of metrics defaults to using Weights and Biases. If this is your first time using WandB, you will be prompted to login/set up an account via the command line. WandB logging can be turned off via `wandb=False`. Model weights and replay buffer is, by default, stored under a `./storage/...` storage folder, but this can be customised as desired.

By default, the 12M parameter model is used, with configuration existing for all other model sizes given in the Dreamer V3 Nature paper.

## Structure
Training and evaluation are driven by the `Dreamer` class in the `./dreamer/dreamer.py` file. This takes the config and builds all relevant sub-classes and modules.

The World Model and Agent Behaviour (actor/critic) are separated into two files `./dreamer/world_model.py` and `./dreamer/behaviour.py`. Each of these are built upon the modules given in `./dreamer/networks/` which contain the key neural network modules, along with the distributions given in `./dreamer/distributions/`.

Environment wrappers are stored in `./dreamer/envs/`.

Evertyhing should hopefully be easily hackable/extendable, and effort has been made to type hint and document the majority of functions and classes.

## Environments
The following environments are supported:

- Deep Mind Control Suite (tested for visual only, but vector observations should also be handled, currently untested)
- MiniGrid (by default uses the FullyObservable RGB wrapper for obserations, but should be easy to customise)
- Crafter
- Atari

Adding new environments should be straightfoward. Simply create a wrappper, following the template used for the existing environments, and add a config file under `dreamer/config/env/{your_env}.yaml`.

## Missing features
The vast majority of the tips and tricks used in the official Dreamer V3 implementation have been implemented here. However, there are still some missing features that may or may not be added in the future, which can cause some difference to the official JAX version:

- Parallel environments. This is the big one in terms of speed for any environments which are slow to step. At the moment, there is a single training environment for simplicity.
- Replay Context. The official Dreamer V3 uses a 'replay context' length of 1 to 'warm-up' the latent state on the replay buffer sample used to train the world model. Essentially, since a replay buffer sample can start in the middle of the episode, this context uses the previous time step's observation to create the initial latent state. Without this, the initial latent state is simply set to a zero vector, which is what this repository does always.
- Replay Loss. The official implementation uses an additional Critic loss that is trained on samples from the replay buffer as well as imagined states. This is not yet implemented here.
- Optimiser. The official implementation uses adaptive gradient clipping combined with LaProp. We instead currently opt for a simpler Adam-based optimiser at the moment.

If you think we've missed any key functionality differences, do open an issue and let us know, although there is no guarantee that we will have time to implement any of these.

## Acknowledgements
This implementation would not be possible without the fantastic existing PyTorch port: [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) by NM512.

## License
This codebase is licensed under the open-source MIT License.
