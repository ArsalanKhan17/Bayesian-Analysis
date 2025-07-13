# Bayesian Insurance Analysis

A comprehensive Bayesian statistical analysis of insurance charges using R and the `rstanarm` package. This project explores the relationship between demographic/health factors and insurance costs through multiple hierarchical Bayesian models.

## Overview

This analysis implements four progressively complex Bayesian regression models to predict insurance charges:

1. **Model 1**: Basic Bayesian linear regression
2. **Model 2**: Adds interaction terms (age×smoking, BMI×smoking)  
3. **Model 3**: Log-transformed response variable
4. **Model 4**: Combined log transformation with interactions (most comprehensive)

## Dataset

The analysis uses an insurance dataset with the following variables:
- `age`: Age of the insured
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of dependents
- `smoker`: Smoking status (yes/no)
- `region`: Geographic region (northeast, northwest, southeast, southwest)
- `charges`: Insurance charges (target variable)

## Key Features

### Model Comparison
- Leave-one-out cross-validation (LOO-CV) for robust model selection
- Bayesian R-squared comparisons across all models
- Information criteria analysis

### Comprehensive Diagnostics
- MCMC convergence diagnostics (Rhat statistics, trace plots)
- Posterior predictive checks
- Residual analysis and normality tests
- Scale-location plots for heteroscedasticity assessment

### Effect Analysis
- Posterior summaries with credible intervals
- Probability statements for effect directions
- Multiplicative effects and percentage increases (for log-scale models)
- Interaction effect quantification

### Uncertainty Quantification
- Full posterior predictive distributions for new observations
- 95% credible intervals for all parameters
- Uncertainty propagation through model predictions

## Requirements

```r
# Required R packages
library(rstanarm)     # Bayesian modeling
library(tidyverse)    # Data manipulation
library(bayesplot)    # Bayesian visualization
library(ggplot2)      # Plotting
library(posterior)    # Posterior analysis
library(loo)          # Model comparison
```

## Usage

Run the complete analysis:

```r
source("bayesian_insurance_analysis.R")
```

The script performs:
1. Exploratory data analysis with visualizations
2. Sequential model fitting with MCMC sampling
3. Model comparison using LOO cross-validation
4. Posterior analysis and effect estimation
5. Predictive modeling with uncertainty quantification
6. Comprehensive model validation

## Model Architecture

### Priors
- **Models 1-2**: Weakly informative Normal(0, 10000) priors
- **Models 3-4**: More informative Normal(0, 10) priors for log-scale

### MCMC Configuration
- 4 parallel chains
- 2000 iterations per chain
- Fixed seed (123) for reproducibility

### Key Findings

The analysis reveals:
- Strong association between smoking status and insurance charges
- Age and BMI effects on insurance costs
- Significant interactions between smoking and other risk factors
- Improved model fit with log transformation and interactions

## Visualizations

The analysis generates comprehensive visualizations organized into several categories:

### Exploratory Data Analysis
```r
# Distribution of target variable
p1 <- ggplot(insurance_data, aes(x = charges)) +
  geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7) +
  labs(title = "Distribution of Insurance Charges")

# Charges by smoking status  
p2 <- ggplot(insurance_data, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Charges by Smoking Status")

# Age vs charges relationship
p3 <- ggplot(insurance_data, aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Age vs Charges by Smoking Status")
```

### Bayesian Model Diagnostics
```r
# Posterior predictive checks
pp_check(model1, nreps = 50) + 
  labs(title = "Posterior Predictive Check")

# MCMC trace plots for convergence assessment
mcmc_trace(model1, pars = c("age", "sexmale", "bmi", "smokeryes"))

# Posterior distribution areas
mcmc_areas(model1, pars = c("age", "sexmale", "bmi", "smokeryes")) +
  labs(title = "Posterior Distributions of Key Parameters")
```

### Parameter Estimation
```r
# Coefficient plots with credible intervals
plot(model1, pars = c("age", "sexmale", "bmi", "children", "smokeryes")) +
  labs(title = "Coefficient Estimates with Credible Intervals")

# Model 2 with interaction terms
mcmc_areas(model2, pars = c("age", "sexmale", "bmi", "smokeryes", 
                           "age:smokeryes", "bmi:smokeryes")) +
  labs(title = "Model 2: Posterior Distributions with Interactions")
```

### Model Validation
```r
# Residual analysis
plot(fitted_values, residuals, main = "Residuals vs Fitted Values")
qqnorm(residuals, main = "Normal Q-Q Plot of Residuals")

# Enhanced diagnostics for log-transformed models
par(mfrow = c(2, 2))
plot(fitted_values3, residuals3, main = "Model 3: Residuals vs Fitted (Log Scale)")
qqnorm(residuals3, main = "Model 3: Normal Q-Q Plot")
plot(fitted_original, residuals3, main = "Model 3: Residuals vs Fitted (Original Scale)")
hist(residuals3, main = "Model 3: Residual Distribution", col = "lightblue")
```

### Advanced Model 4 Diagnostics
```r
# Comprehensive residual analysis
plot(fitted_values4, residuals4, main = "Model 4: Residuals vs Fitted (Log Scale)")
plot(fitted_values4, sqrt(abs(residuals4)), main = "Model 4: Scale-Location Plot")

# Final posterior predictive check
pp_check(model4, nreps = 50) + labs(title = "Model 4: Posterior Predictive Check")
```

**Visualization Highlights:**
- **Exploratory plots** reveal right-skewed charge distribution and clear smoking effects
- **Trace plots** confirm MCMC convergence across all parameters
- **Posterior areas** show parameter uncertainty and effect magnitudes  
- **Residual diagnostics** validate model assumptions and identify improvements
- **Interaction visualizations** demonstrate complex relationships between predictors

## Project Structure

```
├── bayesian_insurance_analysis.R    # Main analysis script
├── insurance.csv                    # Dataset
├── Practice.Rproj                   # R project file
└── README.md                        # This file
```

## Technical Notes

- Models 3 and 4 use log-transformed responses requiring back-transformation via `exp()` for interpretation
- Large model objects may require significant memory for MCMC sampling
- All visualizations use `ggplot2` with `theme_minimal()` for clean presentation
- The analysis includes comprehensive residual diagnostics for all models