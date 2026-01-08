# Using Boltz on the cluster 

## Prepare the directories on the cluster
```bash
    export BOLTZ_CACHE=/p/scratch/<project>/.boltz_cache
    export BOLTZ_INPUT=/p/project1/<project>/<Username>/boltz_input
    export BOLTZ_OUTPUTS=/p/project1/<project>/<Username>/boltz_output

    # Make directories if not already existing
    mkdir -p $BOLTZ_CACHE $BOLTZ_INPUT $BOLTZ_OUTPUTS
```
Alternatively a guide on how to setup the cluster can be found here : https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/jsc_basics/

## Prepare the input files

### Create yaml files
    Run the scripts/create_boltz_input.py to create the yaml files for each prediction and store them in the BOLTZ_INPUT folder
```bash
    python scripts/create_boltz_input.py --output /path/to/output --msa /path/to/msa
```
### create msa files
    This is step is only necessary if you have to run the msa process for boltz locally, so no --use_msa_server flag. When running on the compute node one does not have acces to the Internet so this step is necessary. 
    You need to run it in an env. You can use the env for the project or a seperate one.
    Install colabfold[alphafold] in an enviroment 
```bash
    # Install PyTorch (CPU or GPU version)
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    # Or GPU (CUDA 11.8)
    # conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

    # Install ColabFold
    pip install colabfold[alphafold-minus-jax] #for CPU only 
```

    Then inside the env :
```bash 
colabfold_batch <input_sequences>.fasta <output_folder> --msa-only
```
This will produce an output inside the output folder with a3m and pickles files for each protein. Only the a3m files are required for Boltz

### Create an apptainer application.
Apptainer is a tool for running containerized applications, similar to Docker, designed for high-performance computing (HPC) environments. It allows you to package software with all dependencies and run it reproducibly on different systems. Compute nodes usually do not have access
Building the apptainer image on the Cluster may not work and it may have to be done after downloading it locally (and send to the cluster later)

```bash
    apptainer build boltz.sif boltz.def
```
boltz.def is already on git and contains the module that needed to be included for the env. while boltz.sif is being generated in the above command and needed for the slurm script

### Schedule an Slurm job

Now all that is left to run a slurm job with sbatch. 
Make sure to modify the sbatch file so that the path to your directories is correct. 
```bash
    sbatch boltz_predict.sbatch
```



