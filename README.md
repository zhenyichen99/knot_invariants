# Knot Invariants

## Intro
This project is an attempt to find connections between hyperbolic and quantum/algebraic knot invariants, using the techniques of deep learning. The hope is to extend the results of [Davies et al.](https://msp.org/gt/2024/28-5/gt-v28-n5-p06-s.pdf)

In the repository, you will find the datasets of knot invariants and the scripts used to generate them. Since the project is still ongoing, I plan to upload the experiment scripts when they are more mature, but I included some initial findings below, as well as the methods for data generation and preprocessing.

## Files
`Regina.zip` contains data from the Regina census, converted to SnapPy readable DT code

`invariants.zip` contains the raw datasets of knot invariants, as well as the preprocessed datasets ready for experiments

`data_gen.py` is the main script used to generate the invariants

`data_util.py` contains various utilities for data preprocessing and sampling 

`model.py` defines a simple MLP and a dropout variant

`train_test.py` defines the train and test loops

`run.py` loads the data and trains the model

## Data Generation
The starting point of our datasets is the [Regina census](https://regina-normal.github.io/data.html), which tabulates all prime knots up to 19 crossings. For this project, I generated invariants for all hyperbolic knots up to 16 corssings, of which there are about 1.7 millions. 

The input invariants for the experiments are hyperbolic invariants. Using the program [SnapPy](https://snappy.computop.org), I computed a whole suite of them, including volume, Chern-Simons invariant, cusp area, cusp translations, systole length and torsion, hyperbolic (adjoint) torsion polynomials. Calculations for these invariants sometimes fail for floating point errors, which can be remedied by SnapPy's high precision manifold class `snappy.ManifoldHP`. However, even the high precision class can fail in some cases. The following table summarizes the failure rate for each invariant.

| input invariant | failure rate (%) |
|--|--|
| volume | 0.0 |
| Chern-Simons | 0.0 |
| cusp area | 0.06 |
| cusp translations | 0.21 |
| hyperbolic torsion | 13.83 |

The systole length and torsion take a long time to calculate, so they are not yet inluded in the datasets but will be in a future update. The hyperbolic adjoint torsion had a very high failure rate (61%) on knots with at most 12 crossings, so I decided to cut it out.

The target invariants are of a more algebraic nature. So far, I have experimented with the Jones polynomial and the sigature, both of which SnapPy had no problem computing.
| target invariant | failure rate (%) |
|--|--|
| signature | 0.0 |
| Jones polynomial | 0.0 |

## Data Preprocessing
Some entries of the Chern-Simons invariant are very small (<1e-10), which shouldn't happen and suggests the true values are zero, so we set them to zero.

The next step is dropping all knots with a missing invariant (due to calculation failure). Since the hyperbolic torsion polynomial failed significantly more often than the other invariants, I made three input datasets, tabulated below.

| dataset | input features |
|--|--|
| input set 1 | volume, Chern-Simons, cusp area, and cusp translations |
| input set 2 | input set 1 + hyperbolic torsion |
| input set 3 | hyperbolic torsion |

All input features are then normalized by subtracting the mean and dividing by the variance. The input data are then paired with the target invariants according to the knots' DT code.

Finally, the experiement-ready datasets are split according to the ratio train/val/test = 90/5/5.

## Experiment 0: Signature
To start, I repeated the work of [Davies et al.](https://msp.org/gt/2024/28-5/gt-v28-n5-p06-s.pdf) on my datasets, so as to verify the data generation and preprocessing steps didn't go terribly wrong. Since their saliency analysis suggests the cusp translations are the most relevant features for predicting the signature, I picked input set 1 for this experiment. Here are the results:

| classifier | test accuracy (%) |
|--|--|
| ZeroR baseline | 30.3 |
| depth-3 width-256 MLP | 81-84 |

Our test accuracy is slightly higher than their 78%, which is unsurprising given that they used a bigger and more random dataset.

## Experiment 1: Jones constant
### Experiment 1.1
In this experiment, I tried to predict the constant term of the Jones polynomial from input set 1. The ZeroR classifier gives a baseline accuracy of 10.5%. I trained an MLP with 3 hidden layers of width 2048. The training loss never dropped below 3, and test accuracy stayed between 11-14%. Doubling the layer width did not improve the results. 

| classifier | test accuracy (%) |
|--|--|
| ZeroR baseline | 10.5 |
| depth-3 width-2048 MLP | 11-14 |
| depth-3 width-4096 MLP | 11-14 |

Since the accuracy isn't significantly better than the baseline, this experiment suggests the selected hyperbolic invariants might not contain sufficient information to compute the Jones polynomial.

### Experiment 1.2 
This experiment repeats the above but uses input set 2 instead. The MLPs performed slightly better this time, but still not enough to suggest a meaningful connection.

| classifier | test accuracy (%) |
|--|--|
| ZeroR | 11.4 |
| depth-3 width-2048 MLP | 17-21 |
| depth-3 width-4096 MLP | 17-21 |

## Experiment 2: Alexander constant
### Experiment 2.1
This experiment trains a depth-3 width-400 MLP to predict the constant term of the Alexander polynomial from input set 1. The model performed worse than the baseline, suggesting the lack of a statistical connection.

| classifier | test accuracy (%) |
|--|--|
| ZeroR | 40.3 |
| depth-3 width-2048 MLP | 35-39 |

### Experiment 2.2
This experiment repeats the above with input set 2. The model performed significantly better than the baseline, which gives us confidence that the Alexander constant is indeed related to the hyperbolic invariants. Adding dropout layers at a rate of 0.2 improved the accuracy slightly.

| classifier | test accuracy (%) |
|--|--|
| ZeroR | 40.1 |
| depth-3 width-2048 MLP | 80-82 |
| MLP with 0.2 dropout rate | 81-83 |

### Experiment 2.3
In light of the performance difference between experiments 2.1 and 2.2, it's very natural to suspect that the hyperbolic torsion polynomial alone should be enough to replicate the results of experiment 2.2, since the other invariants don't seem to help at all. Indeed, running the experiment again with input set 3 did give similar results:

| classifier | test accuracy (%) |
|--|--|
| ZeroR | 40.1 |
| depth-3 width-2048 MLP | 80-82 |
| MLP with 0.2 dropout rate | 80-82 |

### Conclusion
Experiment 2 tells us the hyperbolic invariants are able to predict the Alexander constant to a reasonable accuracy. Moreover, the most salient feature is the hyperbolic torsion polynomial. 

### Next Steps
I plan on using attribution techiques to probe the learned model from experiment 2.3. The goal is to extract a formula or inequality expressing the Alexander constant in terms of the coefficients of the hyperbolic torsion polynomial. 
