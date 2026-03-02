# Demographic Microsimulator (DEMOS)

[![Docs](https://github.com/NREL/DEMOS/actions/workflows/docs.yml/badge.svg)](https://nrel.github.io/DEMOS/)

## Overview
The Demographic Microsimulator (DEMOS) is an agent-based simulation framework used to model the evolution of population demographic characteristics and lifecycle events, such as education attainment, marital status, and other key transitions. DEMOS modules are designed to capture the interdependencies between short-term and long-term lifecycle events, which are often influential in downstream transportation and land-use modeling.

A key feature of DEMOS is its ability to track changes in an agent’s demographic status from year *t* to year *t + 1*. This structure allows the model to evolve populations over any user-defined time horizon. As a result, DEMOS is well suited for analyzing medium- and long-term transportation-related decisions, including household vehicle transactions (e.g., purchasing, selling, or replacing vehicles) and work location choices.
Core features of DEMOS include the modeling of more than ten lifecycle events, behaviorally realistic patterns informed by long-running panel data, explicit representation of interdependencies among lifecycle processes, and a flexible, modular simulation architecture.

A technical memorandum describing DEMOS is available [here](./DEMOS_Technical_Memo.pdf). The memorandum provides an overview of the framework’s functionality, model structure, input and output data, and its applications in transportation planning and broader policy analysis contexts. Interested readers are also encouraged to consult the paper listed below for additional details on the DEMOS methodology.


*Sun, Bingrong, Shivam Sharda, Venu M. Garikapati, Mohamed Amine Bouzaghrane, Juan Caicedo, Srinath Ravulaparthy, Isabel Viegas de Lima, Ling Jin, C. Anna Spurlock, and Paul Waddell. "Demographic Microsimulator for Integrated Urban Systems: Adapting Panel Survey of Income Dynamics to Capture the Continuum of Life." Transportation Research Record (2025): 03611981251333339.*

## Usage

### Docker Compose (recommended)
The latest docker image for demos is stored in `ghcr.io/NatLabRockies/demos:latest`. The input data and configuration file are fed to the container through volumes ([more info about Docker volumes](https://docs.docker.com/engine/storage/volumes/)). We provide a `docker-compose` workflow that can be used to make the process of mounting volumes easier.

#### Clone this repository
By cloning this repository you download the configuration and data for an example run of DEMOS.

Run the following command in the Terminal App (MacOS) or Command Prompt/PowerShell (Windows):
```bash
git clone https://github.com/NatLabRockies/DEMOS.git

# Move into the project folder
cd DEMOS

# This folder contains (among other files) a data and configuration folder
# as well as a docker-compose.yml file
```

Make sure you have [Docker](https://docs.docker.com/desktop/) and [Docker Compose](https://docs.docker.com/compose/install/) installed. Now you can run docker as follows:

```bash
docker compose up
```
#### IMPORTANT for MacOS and Windows users
> Docker imposes a global limit on how much RAM containers can allocate. DEMOS easily surpases those limits, so in order to run DEMOS in Docker, users need to access the Docker Desktop GUI and `Preferences → Resources → Memory → Increase it (at least 16-20gb)`. The amount of memory required to run DEMOS will primarily depend on the size of the input data.

Documentation for custom data requirements, configuration and overall functionality of demos can be found [in the Docs](https://nrel.github.io/DEMOS/).

## Other ways to run DEMOS

### Running from Source

If you prefer to create your own Python environment and run the Python code directly (for debugging or enhancing DEMOS capabilities), you can do so following these instructions:

1. Clone this repository
	```
	git clone https://github.com/NatLabRockies/DEMOS.git
	```

1. Create a virtual environment.

	**If using conda**, prefer the provided `.lock` files
	```
	conda create --name demos-env --file conda-{system}.lock
	```

	Alternatively, create a `python 3.10` environment and install dependencies
	```
	conda create -n demos-env python=3.10
	conda activate demos-env
	pip install .
	```

## Contact
If you have questions, suggestions, or are interested in collaborating, please feel free to reach out or open an issue.  
Bingrong Sun: Bingrong.Sun@nlr.gov  
Shivam Sharda: Shivam.Sharda@nlr.gov  
Venu Garikapati: Venu.Garikapati@nlr.gov  
