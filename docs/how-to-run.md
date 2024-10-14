# Evaluation setup
We've provided one example on how we can run the evaluation using `slurm` and `singularity`, and one example using only `docker`. The `slurm`-based evaluation script will launch each of the scenarios as a seperate job, resulting in a total of X jobs. Each job will run the simulation 50 times, with slight perturbations to the initial conditions. The resutlts will then be stored in the `neuro-ncap/outputs` directory. For the `docker`-based evaluation, the scenarios will be run sequentially on a single machine. However, the results will be the same.

## Folder structure
In the project we have three seperate repositories, namely the `rendering (neurad-studio)`, `simulation (neuro-ncap)` and `model (UniAD/VAD/<your-model>)` repositories. In our example we assume that each of these are cloned into the same parent directory. The folder structure should look like this:

```
.
├── neurad-studio
│   ├── checkpoints
│   |   ├── scene-0103.pth
│   |   ├── ...
│   ├── neurad-studio.sif (if running using singularity)
│   ├── ...
├── neuro-ncap
│   ├── neuro-ncap.sif (if running using singularity)
│   ├── ...
├── UniAD (or VAD, or your own model)
|   ├── checkpoints
│   |   ├── uniad_base_e2e.pth
│   ├── uniad.sif (if running using singularity)
│   ├── ...
```

Given that you have already cloned this repository, and currently standing in the root of the `neuro-ncap` repository, you can clone the other repositories by running the following commands:

```bash
git clone https://github.com/georghess/neurad-studio.git ../neurad-studio
git clone https://github.com/wljungbergh/UniAD.git ../UniAD
# (and download weights according to UniAD README)
```

To run use your own model, simply have your model repository in the same parent directory as the other repositories. The model repository should contain an `inference/server.py`, similar to the one in the UniAD repository. For more information on how to implement the server, please refer to the [Model Node](../docs/OVERVIEW.md#model-node) section in the overview.

The reason for this is that we dont want to enforce the enviorment from either of the repositories on the other. This way we can easily switch out the model repository with your own model, without having to change the other repositories.

## Prerequisites

### NeuRAD checkpoints
The `neurad-studio` repository requires the NeuRAD checkpoints to be downloaded. This can be done by running the `download_neurad_weights.sh` script in the `scripts/downloads` folder: `bash scripts/downloads/download_neurad_weights.sh`.

### Docker images
For each of the projects (`rendering`, `ncap`, `model`), we provide Docker files. These can be used to build a singularity file. This can be done by running the following command in the root of each repository:
```bash
docker build -t <image-name>:latest -f docker/Dockerfile .
```
Note that if you are using your own model, you have to build a custom docker image for that repository with the appropriate depencencies.

### Singularity files (only required for the `slurm`-based running option)
To build a singularity file from each docker image, run the following for each project:
```bash
singularity build <image-name>.sif docker-daemon://<image-name>:latest
```

## Running the project with singularity and slurm (recommended)
**This will run the evaluation on a slurm cluster, and each scenario will be run as a seperate job. As they will be ran in parrallel, the total evaluation time can take less than an hour (assuming enough available GPUs)**

To run the evaluation, edit the variables in the top of the `slurm_singularity_run_eval.sh` script to match your setup. Then simply run the following command from the root of the `neuro-ncap` repository:

```bash
./slurm_singularity_run_eval.sh
```

## Running the project with docker on a single machine
**This will run the evaluation sequentially on a single machine using only docker. This is more suitable when you dont have access to a compute cluser, but rather a single machine with a (powerful) GPU. Note that this can take >24h, and has been tested and verified on a NVIDIA RTX 3060 (12GB)**

To run the evaluation, edit the variables in the top of the `single_machine_docker_run_eval.sh` script to match your setup. Then simply run the following command from the root of the `neuro-ncap` repository:

```bash
./single_machine_docker_run_eval.sh
```

## Aggregating the results

Once all jobs have finished, we have to aggregate the results across all scenarios. This can be done by running the following command:

```bash
python3 scripts/aggregate_results.py outputs/<date>
```

This will print the NCAP score table to the console, and save the results in a JSON file in the `neuro-ncap/outputs/<date>` directory. `<date>` is the date of the evaluation, and is set by the slurm/docker running script.
