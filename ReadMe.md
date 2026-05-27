# Reproducible Baselines for ML Research on Sensitive Data

This repository is the official implementation of the benchmark conducted in [Reproducible Baselines for ML Research on Sensitive Data](TODO). 


## Requirements

Clone the Project:
```clone
git clone ...
cd AutoBaseline
```

We prepared a shell script to set up a virtual environment, you can run it with:

```setup
./hpc/venv_setup.sh
```

Note: You might need to adjust the script to confirm with your available modules.


## Benchmark

To perform the benchmark, run the following:

```train
sbatch ./hpc/OpenML-CTR23-HEBO.sh
```
Note: You might need to adjust the configuration of the slurm script

This creates a sqlite database containing the results of the benchmark. 
Further, it also creates several smaller databases that cache the results of individual trials to avoid write conflicts on the main database due to massive parallelism.
To add those temporary databases to the main database, run:

```merge
sbatch ./hpc/cache2db.sh
```



## Analysis and Visualization

To analyze the results of the benchmark, we provide a jupyter notebook at `analysis\performance_comparison.ipynb` that contains the code to analyze the results and produce the figures in the paper.


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 


### Disclaimer/TODO
The analysis was conducted on older data when we didn't track experiment IDs this shouldn't change the experiment results.
However, the analysis is deprecated and needs to be updated else it will raise errors