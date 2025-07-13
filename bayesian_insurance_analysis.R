# Bayesian Analysis of Insurance Data using rstanarm
# =================================================

# Load required libraries
library(rstanarm)
library(tidyverse)
library(bayesplot)
library(ggplot2)
library(posterior)
library(loo)

# Load the data
insurance_data <- read.csv("insurance.csv")

# Data exploration
cat("Dataset dimensions:", dim(insurance_data), "\n")
cat("First few rows:\n")
head(insurance_data)

cat("\nSummary statistics:\n")
summary(insurance_data)

cat("\nMissing values:\n")
sapply(insurance_data, function(x) sum(is.na(x)))

# Convert categorical variables to factors
insurance_data$sex <- as.factor(insurance_data$sex)
insurance_data$smoker <- as.factor(insurance_data$smoker)
insurance_data$region <- as.factor(insurance_data$region)

# Exploratory visualizations
p1 <- ggplot(insurance_data, aes(x = charges)) +
  geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7) +
  labs(title = "Distribution of Insurance Charges", x = "Charges", y = "Frequency") +
  theme_minimal()

p2 <- ggplot(insurance_data, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Charges by Smoking Status", x = "Smoker", y = "Charges") +
  theme_minimal()

p3 <- ggplot(insurance_data, aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Age vs Charges by Smoking Status", x = "Age", y = "Charges") +
  theme_minimal()

print(p1)
print(p2)
print(p3)

# =================================================
# Bayesian Linear Regression with rstanarm
# =================================================

# Model 1: Simple linear regression
cat("\n=== Model 1: Simple Bayesian Linear Regression ===\n")

# Set seed for reproducibility
set.seed(123)

# Fit Bayesian linear regression
model1 <- stan_glm(charges ~ age + sex + bmi + children + smoker + region,
                   data = insurance_data,
                   family = gaussian(),
                   prior = normal(0, 10000),  # weakly informative priors
                   prior_intercept = normal(0, 10000),
                   chains = 4,
                   iter = 2000,
                   seed = 123)

# Model summary
print(model1)

# =================================================
# Model Diagnostics
# =================================================

cat("\n=== Model Diagnostics ===\n")

# Check convergence
print(model1$stanfit)

# Posterior predictive checks
pp_check(model1, nreps = 50) + 
  labs(title = "Posterior Predictive Check")

# Trace plots
mcmc_trace(model1, pars = c("age", "sexmale", "bmi", "smokeryes"))

# Posterior intervals
posterior_interval(model1, prob = 0.95)

# =================================================
# Model Comparison and Selection
# =================================================

cat("\n=== Model Comparison ===\n")

# Model 2: With interaction terms
model2 <- stan_glm(charges ~ age + sex + bmi + children + smoker + region + 
                   age:smoker + bmi:smoker,
                   data = insurance_data,
                   family = gaussian(),
                   prior = normal(0, 10000),
                   prior_intercept = normal(0, 10000),
                   chains = 4,
                   iter = 2000,
                   seed = 123)

# Model 3: Log-transformed response
model3 <- stan_glm(log(charges) ~ age + sex + bmi + children + smoker + region,
                   data = insurance_data,
                   family = gaussian(),
                   prior = normal(0, 10),
                   prior_intercept = normal(0, 10),
                   chains = 4,
                   iter = 2000,
                   seed = 123)

# Model 4: Both interactions AND log transformation
model4 <- stan_glm(log(charges) ~ age + sex + bmi + children + smoker + region +
                     age:smoker + bmi:smoker,
                   data = insurance_data,
                   family = gaussian(),
                   prior = normal(0, 10),
                   prior_intercept = normal(0, 10),
                   chains = 4,
                   iter = 2000,
                   seed = 123)


# Compare models using LOO
loo1 <- loo(model1)
loo2 <- loo(model2)
loo3 <- loo(model3)
loo4 <- loo(model4)

print(loo1)
print(loo2)
print(loo3)
print(loo4)

# Model comparison
loo_compare(loo1, loo2, loo3)

# Model 3 and 4 (log-normal)
loo_compare(loo3, loo4)

# =================================================
# Posterior Analysis
# =================================================

cat("\n=== Posterior Analysis ===\n")

# Extract posterior draws
posterior_draws <- as.matrix(model1)

# Posterior summaries
posterior_summary <- summarise_draws(posterior_draws)
print(posterior_summary)

# Plot posterior distributions
mcmc_areas(model1, pars = c("age", "sexmale", "bmi", "smokeryes")) +
  labs(title = "Posterior Distributions of Key Parameters")

# Coefficient plots
plot(model1, pars = c("age", "sexmale", "bmi", "children", "smokeryes")) +
  labs(title = "Coefficient Estimates with Credible Intervals")

# Posterior analysis for Model 2
cat("\n=== Model 2 Posterior Analysis ===\n")

# Extract posterior draws for Model 2
posterior_draws2 <- as.matrix(model2)

# Posterior summaries for Model 2
posterior_summary2 <- summarise_draws(posterior_draws2)
print(posterior_summary2)

# Plot posterior distributions for Model 2
mcmc_areas(model2, pars = c("age", "sexmale", "bmi", "smokeryes", "age:smokeryes", "bmi:smokeryes")) +
  labs(title = "Model 2: Posterior Distributions with Interactions")

# Model 4

cat("\n=== Posterior Analysis ===\n")

# Extract posterior draws
posterior_draws <- as.matrix(model4)

# Posterior summaries
posterior_summary <- summarise_draws(posterior_draws)
print(posterior_summary)

# Plot posterior distributions
mcmc_areas(model4, pars = c("age", "sexmale", "bmi", "smokeryes")) +
  labs(title = "Posterior Distributions of Key Parameters")

# Coefficient plots
plot(model4, pars = c("age", "sexmale", "bmi", "children", "smokeryes")) +
  labs(title = "Coefficient Estimates with Credible Intervals")


# =================================================
# Predictions and Uncertainty Quantification
# =================================================

cat("\n=== Predictions ===\n")

# Posterior predictions for new data
new_data <- data.frame(
  age = c(25, 45, 65),
  sex = factor(c("male", "female", "male"), levels = c("female", "male")),
  bmi = c(25, 30, 35),
  children = c(0, 2, 1),
  smoker = factor(c("no", "no", "yes"), levels = c("no", "yes")),
  region = factor(c("southwest", "northeast", "southeast"), 
                  levels = c("northeast", "northwest", "southeast", "southwest"))
)

# Generate posterior predictions
posterior_preds <- posterior_predict(model1, newdata = new_data)

# Summary of predictions
pred_summary <- apply(posterior_preds, 2, function(x) {
  c(mean = mean(x), 
    median = median(x),
    lower = quantile(x, 0.025),
    upper = quantile(x, 0.975))
})

colnames(pred_summary) <- paste0("Person_", 1:3)
print(pred_summary)

# =================================================
# Predictions for Model 3 (Log-transformed)
# =================================================

cat("\n=== Model 3 Predictions ===\n")

# Same new data as before
new_data <- data.frame(
  age = c(25, 45, 65),
  sex = factor(c("male", "female", "male"), levels = c("female", "male")),
  bmi = c(25, 30, 35),
  children = c(0, 2, 1),
  smoker = factor(c("no", "no", "yes"), levels = c("no", "yes")),
  region = factor(c("southwest", "northeast", "southeast"),
                  levels = c("northeast", "northwest", "southeast", "southwest"))
)

# Generate posterior predictions on LOG scale
posterior_preds_log <- posterior_predict(model3, newdata = new_data)

# Transform back to DOLLAR scale
posterior_preds_dollars <- exp(posterior_preds_log)

# Summary of predictions in dollars
pred_summary3 <- apply(posterior_preds_dollars, 2, function(x) {
  c(mean = mean(x),
    median = median(x),
    lower = quantile(x, 0.025),
    upper = quantile(x, 0.975))
})

colnames(pred_summary3) <- paste0("Person_", 1:3)
print(pred_summary3)

# =================================================
# Predictions for Model 4 (Log-transformed and Interaction Terms)
# =================================================

cat("\n=== Model 4 Predictions ===\n")

# Same new data as before
new_data <- data.frame(
  age = c(25, 45, 65),
  sex = factor(c("male", "female", "male"), levels = c("female", "male")),
  bmi = c(25, 30, 35),
  children = c(0, 2, 1),
  smoker = factor(c("no", "no", "yes"), levels = c("no", "yes")),
  region = factor(c("southwest", "northeast", "southeast"),
                  levels = c("northeast", "northwest", "southeast", "southwest"))
)

# Generate posterior predictions on LOG scale
posterior_preds_log <- posterior_predict(model4, newdata = new_data)

# Transform back to DOLLAR scale
posterior_preds_dollars <- exp(posterior_preds_log)

# Summary of predictions in dollars
pred_summary4 <- apply(posterior_preds_dollars, 2, function(x) {
  c(mean = mean(x),
    median = median(x),
    lower = quantile(x, 0.025),
    upper = quantile(x, 0.975))
})

colnames(pred_summary4) <- paste0("Person_", 1:3)
print(pred_summary4)


# =================================================
# Effect Analysis
# =================================================

cat("\n=== Effect Analysis ===\n")

# Smoking effect analysis
smoking_effect <- posterior_draws[, "smokeryes"]
cat("Smoking effect (posterior mean):", mean(smoking_effect), "\n")
cat("95% Credible Interval:", quantile(smoking_effect, c(0.025, 0.975)), "\n")
cat("Probability that smoking increases costs:", mean(smoking_effect > 0), "\n")

# Age effect per year
age_effect <- posterior_draws[, "age"]
cat("Age effect per year (posterior mean):", mean(age_effect), "\n")
cat("95% Credible Interval:", quantile(age_effect, c(0.025, 0.975)), "\n")

# BMI effect
bmi_effect <- posterior_draws[, "bmi"]
cat("BMI effect (posterior mean):", mean(bmi_effect), "\n")
cat("95% Credible Interval:", quantile(bmi_effect, c(0.025, 0.975)), "\n")

# =================================================
# Effect Analysis for Model 3 (Log Scale)
# =================================================

cat("\n=== Model 3 Effect Analysis ===\n")

# Extract posterior draws for Model 3
posterior_draws3 <- as.matrix(model3)

# Smoking effect analysis (on log scale)
smoking_effect_log <- posterior_draws3[, "smokeryes"]
cat("Smoking effect on log scale (posterior mean):", mean(smoking_effect_log), "\n")
cat("95% Credible Interval (log scale):", quantile(smoking_effect_log, c(0.025, 0.975)), "\n")

# Transform to multiplicative effect (original scale)
smoking_multiplier <- exp(smoking_effect_log)
cat("Smoking multiplier (posterior mean):", mean(smoking_multiplier), "\n")
cat("95% Credible Interval (multiplier):", quantile(smoking_multiplier, c(0.025, 0.975)), "\n")

# Percentage increase
smoking_percent <- (smoking_multiplier - 1) * 100
cat("Smoking percentage increase (posterior mean):", mean(smoking_percent), "%\n")
cat("95% Credible Interval (percentage):", quantile(smoking_percent, c(0.025, 0.975)), "%\n")
cat("Probability that smoking increases costs:", mean(smoking_effect_log > 0), "\n")

# Age effect per year (log scale)
age_effect_log <- posterior_draws3[, "age"]
cat("\nAge effect on log scale (posterior mean):", mean(age_effect_log), "\n")
age_multiplier <- exp(age_effect_log)
age_percent <- (age_multiplier - 1) * 100
cat("Age percentage increase per year:", mean(age_percent), "%\n")
cat("95% Credible Interval (percentage):", quantile(age_percent, c(0.025, 0.975)), "%\n")

# BMI effect (log scale)
bmi_effect_log <- posterior_draws3[, "bmi"]
cat("\nBMI effect on log scale (posterior mean):", mean(bmi_effect_log), "\n")
bmi_multiplier <- exp(bmi_effect_log)
bmi_percent <- (bmi_multiplier - 1) * 100
cat("BMI percentage increase per point:", mean(bmi_percent), "%\n")
cat("95% Credible Interval (percentage):", quantile(bmi_percent, c(0.025, 0.975)), "%\n")


# =================================================
# Effect Analysis for Model 4 (Log Scale)
# =================================================

cat("\n=== Model 4 Effect Analysis ===\n")

# Extract posterior draws for Model 4
posterior_draws4 <- as.matrix(model4)

# Smoking effect analysis (on log scale)
smoking_effect_log <- posterior_draws4[, "smokeryes"]
cat("Smoking effect on log scale (posterior mean):", mean(smoking_effect_log), "\n")
cat("95% Credible Interval (log scale):", quantile(smoking_effect_log, c(0.025, 0.975)), "\n")

# Transform to multiplicative effect (original scale)
smoking_multiplier <- exp(smoking_effect_log)
cat("Smoking multiplier (posterior mean):", mean(smoking_multiplier), "\n")
cat("95% Credible Interval (multiplier):", quantile(smoking_multiplier, c(0.025, 0.975)), "\n")

# Percentage increase
smoking_percent <- (smoking_multiplier - 1) * 100
cat("Smoking percentage increase (posterior mean):", mean(smoking_percent), "%\n")
cat("95% Credible Interval (percentage):", quantile(smoking_percent, c(0.025, 0.975)), "%\n")
cat("Probability that smoking increases costs:", mean(smoking_effect_log > 0), "\n")

# Age effect per year (log scale)
age_effect_log <- posterior_draws4[, "age"]
cat("\nAge effect on log scale (posterior mean):", mean(age_effect_log), "\n")
age_multiplier <- exp(age_effect_log)
age_percent <- (age_multiplier - 1) * 100
cat("Age percentage increase per year:", mean(age_percent), "%\n")
cat("95% Credible Interval (percentage):", quantile(age_percent, c(0.025, 0.975)), "%\n")

# BMI effect (log scale)
bmi_effect_log <- posterior_draws4[, "bmi"]
cat("\nBMI effect on log scale (posterior mean):", mean(bmi_effect_log), "\n")
bmi_multiplier <- exp(bmi_effect_log)
bmi_percent <- (bmi_multiplier - 1) * 100
cat("BMI percentage increase per point:", mean(bmi_percent), "%\n")
cat("95% Credible Interval (percentage):", quantile(bmi_percent, c(0.025, 0.975)), "%\n")

# Interaction effects
age_smoke_interaction <- posterior_draws4[, "age:smokeryes"]
bmi_smoke_interaction <- posterior_draws4[, "bmi:smokeryes"]

cat("Age-smoking interaction:", mean(age_smoke_interaction), "\n")
cat("BMI-smoking interaction:", mean(bmi_smoke_interaction), "\n")

# =================================================
# Model Validation
# =================================================

cat("\n=== Model Validation ===\n")

# R-squared Bayesian
bayes_R2 <- bayes_R2(model1)
cat("Bayesian R-squared:")
print(summary(bayes_R2))

# Residual analysis
residuals <- residuals(model1)
fitted_values <- fitted(model1)

# Residual plots
plot(fitted_values, residuals, main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

# QQ plot of residuals
qqnorm(residuals, main = "Normal Q-Q Plot of Residuals")
qqline(residuals, col = "red")

cat("\nBayesian analysis complete!\n")

# =================================================
  # Model 3 Validation
  # =================================================

  cat("\n=== Model 3 Validation ===\n")

  # R-squared Bayesian for Model 3
  bayes_R2_model3 <- bayes_R2(model3)
  cat("Model 3 Bayesian R-squared:")
  print(summary(bayes_R2_model3))

  # Residual analysis for Model 3
  residuals3 <- residuals(model3)
  fitted_values3 <- fitted(model3)

  # Residual plots for Model 3
  par(mfrow = c(2, 2))

  # Plot 1: Residuals vs Fitted (log scale)
  plot(fitted_values3, residuals3,
       main = "Model 3: Residuals vs Fitted (Log Scale)",
       xlab = "Fitted Log(Charges)", ylab = "Residuals")
  abline(h = 0, col = "red", lty = 2)

  # Plot 2: QQ plot of residuals
  qqnorm(residuals3, main = "Model 3: Normal Q-Q Plot")
  qqline(residuals3, col = "red")

  # Plot 3: Residuals vs Fitted (original scale)
  fitted_original <- exp(fitted_values3)
  plot(fitted_original, residuals3,
       main = "Model 3: Residuals vs Fitted (Original Scale)",
       xlab = "Fitted Charges ($)", ylab = "Residuals (Log Scale)")
  abline(h = 0, col = "red", lty = 2)

  # Plot 4: Histogram of residuals
  hist(residuals3, breaks = 30, main = "Model 3: Residual Distribution",
       xlab = "Residuals", col = "lightblue")

  par(mfrow = c(1, 1))

  # Compare R-squared between models
  cat("\n=== Model Comparison ===\n")
  cat("Model 1 R-squared (mean):", round(mean(bayes_R2), 3), "\n")
  cat("Model 3 R-squared (mean):", round(mean(bayes_R2_model3), 3), "\n")
  cat("Difference:", round(mean(bayes_R2_model3) - mean(bayes_R2), 3), "\n")
  
  
# =================================================
# Model 4 Validation
# =================================================
  
  cat("\n=== Model 4 Validation ===\n")
  
  # R-squared Bayesian for Model 4
  bayes_R2_model4 <- bayes_R2(model4)
  cat("Model 4 Bayesian R-squared:")
  print(summary(bayes_R2_model4))
  
  # Residual analysis for Model 4
  residuals4 <- residuals(model4)
  fitted_values4 <- fitted(model4)
  
  # Residual plots for Model 4
  par(mfrow = c(2, 2))
  
  # Plot 1: Residuals vs Fitted (log scale)
  plot(fitted_values4, residuals4,
       main = "Model 4: Residuals vs Fitted (Log Scale)",
       xlab = "Fitted Log(Charges)", ylab = "Residuals")
  abline(h = 0, col = "red", lty = 2)
  
  # Plot 2: QQ plot of residuals
  qqnorm(residuals4, main = "Model 4: Normal Q-Q Plot")
  qqline(residuals4, col = "red")
  
  # Plot 3: Residuals vs Fitted (original scale)
  fitted_original4 <- exp(fitted_values4)
  plot(fitted_original4, residuals4,
       main = "Model 4: Residuals vs Fitted (Original Scale)",
       xlab = "Fitted Charges ($)", ylab = "Residuals (Log Scale)")
  abline(h = 0, col = "red", lty = 2)
  
  # Plot 4: Scale-Location plot
  plot(fitted_values4, sqrt(abs(residuals4)),
       main = "Model 4: Scale-Location Plot",
       xlab = "Fitted Log(Charges)", ylab = "√|Residuals|")
  
  par(mfrow = c(1, 1))
  
  # Compare R-squared across all models
  cat("\n=== R-squared Comparison ===\n")
  cat("Model 1 R-squared (mean):", round(mean(bayes_R2), 3), "\n")
  cat("Model 3 R-squared (mean):", round(mean(bayes_R2_model3), 3), "\n")
  cat("Model 4 R-squared (mean):", round(mean(bayes_R2_model4), 3), "\n")
  
  # Model 4 improvements
  cat("\nModel 4 improvements:\n")
  cat("vs Model 1:", round(mean(bayes_R2_model4) - mean(bayes_R2), 3), "\n")
  cat("vs Model 3:", round(mean(bayes_R2_model4) - mean(bayes_R2_model3), 3), "\n")
  
  # Posterior predictive checks for Model 4
  cat("\n=== Posterior Predictive Checks ===\n")
  pp_check(model4, nreps = 50) +
    labs(title = "Model 4: Posterior Predictive Check")
  
  cat("\nModel 4 validation complete!\n")
  