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

Note that the setup script may need to be adapted to the modules available on your system.


## Benchmark

To run the benchmark, execute:

```train
sbatch ./hpc/OpenML-CTR23-HEBO.sh
```
The SLURM configuration may need to be adjusted to match your compute environment.

The benchmark writes its results to an SQLite database called `results.db`. It also creates several smaller cache databases that store the results of individual trials. These cache databases are used to avoid write conflicts on the main database during massively parallel execution.

To merge the cache databases into the main results database, run:

```merge
sbatch ./hpc/cache2db.sh
```



## Analysis and Visualization

The analysis code is provided in: `analysis\performance_comparison.ipynb`.
This notebook reproduces the benchmark analysis and generates the figures used in the paper.


## Results

> [!Important]
> Note that the results shown here include more models than the paper. In the paper, models that crashed too frequently on the virtual metrology datasets were removed to enable a consistent comparison between the virtual metrology benchmark and CTR23. They are retained here because this repository focuses only on CTR23, where all models can be evaluated.
 

Our comparison shows that AutoGluon outperforms any individual optimized machine learning method on CTR23.

![Critical Difference Diagram](analysis/critical_difference_diagram_ctr.png)



| Model | Norm RMSE | Avg Rank | Precision | Accuracy | Wins vs MeanModel | Wins vs AG/Mean |
| --- | --- | --- | --- | --- | --- | --- |
| MeanModel | 0.79 | 29.20 | 0.04 | 0.14 | -- | -- |
| MedianModel | 0.81 | 30.60 | 0.03 | 0.09 | 0 (3) | 0 (0) |
| Dummy | 0.79 | 29.17 | 0.06 | 0.14 | 0 (5) | 0 (1) |
| RidgeRegressor | 0.56 | 18.91 | 0.12 | 0.47 | **32** (34) | **1** (**3**) |
| Lasso | 0.57 | 21.06 | 0.06 | 0.40 | **32** (34) | **1** (2) |
| LARS | 0.61 | 23.40 | 0.05 | 0.32 | 24 (27) | **1** (2) |
| LassoLars | 0.60 | 20.06 | 0.08 | 0.43 | 31 (33) | **1** (**3**) |
| ElasticNet | 0.60 | 21.31 | 0.06 | 0.39 | **32** (34) | **1** (2) |
| Huber | 0.65 | 22.86 | 0.10 | 0.34 | 27 (33) | **1** (**3**) |
| BayesianRidge | 0.56 | 18.37 | 0.08 | 0.48 | **32** (34) | **1** (**3**) |
| PassiveAggressive | 0.80 | 30.00 | 0.04 | 0.12 | 8 (12) | **1** (1) |
| SGD | 0.77 | 28.71 | 0.04 | 0.15 | 16 (23) | **1** (1) |
| PartialLeastSquares | 0.57 | 20.60 | 0.12 | 0.41 | **32** (34) | **1** (**3**) |
| Tweedie | 0.63 | 23.06 | 0.05 | 0.33 | 24 (26) | 0 (0) |
| KNN | 0.60 | 20.31 | 0.06 | 0.42 | 30 (34) | 0 (0) |
| MLP | 0.65 | 23.60 | 0.06 | 0.31 | 22 (23) | 0 (0) |
| GaussianProcess | 0.55 | 18.63 | 0.07 | 0.47 | 27 (30) | 0 (0) |
| Bagging | 0.58 | 21.60 | 0.05 | 0.38 | 31 (34) | 0 (0) |
| RandomForest | 0.34 | 13.34 | 0.10 | 0.64 | 30 (32) | 0 (0) |
| ExtraTrees | 0.27 | 11.60 | 0.17 | 0.69 | **32** (33) | 0 (**3**) |
| DecisionTree | 0.54 | 20.23 | 0.06 | 0.42 | 30 (32) | 0 (0) |
| M5 | 0.28 | 12.66 | 0.13 | 0.66 | 29 (33) | 0 (**3**) |
| CubistModel | 0.21 | 11.14 | 0.20 | 0.71 | 31 (33) | **1** (2) |
| GradientBoosting | 0.20 | 12.20 | 0.11 | 0.67 | **32** (33) | 0 (1) |
| AdaBoost | 0.34 | 15.03 | 0.08 | 0.59 | 31 (34) | **1** (2) |
| XGBoost | 0.21 | 11.40 | 0.12 | 0.70 | 31 (**35**) | 0 (1) |
| LightGBM | 0.20 | 11.14 | 0.14 | 0.71 | 31 (34) | 0 (1) |
| CatBoost | 0.15 | 9.66 | 0.18 | 0.75 | 31 (34) | 0 (1) |
| AutoGluon(5min) | 0.12 | 7.06 | 0.26 | 0.83 | 31 (33) | -- |
| AutoGluon(10min) | 0.07 | 6.03 | 0.30 | 0.86 | 31 (33) | -- |
| AutoGluon(30min) | 0.04 | 4.40 | 0.48 | 0.92 | **32** (34) | -- |
| AutoGluon(1h) | 0.02 | 4.09 | 0.66 | 0.93 | **32** (34) | -- |
| AutoGluon(2h) | 0.01 | 3.17 | 0.83 | 0.94 | **32** (34) | -- |
| AutoGluon(worst) | 0.52 | 16.34 | 0.08 | 0.54 | **32** (34) | -- |
| AutoGluon(best) | 0.00 | 2.20 | 0.90 | 0.97 | **32** (34) | -- |
| AutoGluon/Mean(best) | **0.00** | **2.09** | **0.90** | **0.97** | **32** (34) | -- |



On average, optimizing the training duration of AutoGluon achieves performance comparable to optimizing the hyperparameters of 28 individual machine learning methods for up to two hours each and selecting the best-performing method, i.e., the virtual best. In contrast to tuning many individual models, adjusting the training duration of AutoGluon requires substantially less machine learning expertise, engineering effort, time investment, and computational resources.

This distinction is practically important because researchers often resort to using the mean model as a simpler baseline. However, the mean model performs, on average, 79% worse than the virtual best. The closest individual contender is CatBoost, which performs 15% worse than the virtual best after tuning.
