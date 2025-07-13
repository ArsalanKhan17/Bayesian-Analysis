# Generate Plots for README
# This script creates and saves key visualizations from the Bayesian analysis

# Load required libraries
library(rstanarm)
library(tidyverse)
library(bayesplot)
library(ggplot2)
library(posterior)
library(loo)

# Load the data
insurance_data <- read.csv("insurance.csv")

# Convert categorical variables to factors
insurance_data$sex <- as.factor(insurance_data$sex)
insurance_data$smoker <- as.factor(insurance_data$smoker)
insurance_data$region <- as.factor(insurance_data$region)

# Create plots directory
if (!dir.exists("plots")) {
  dir.create("plots")
}

# Set seed for reproducibility
set.seed(123)

cat("Generating exploratory plots...\n")

# 1. Exploratory Data Analysis Plots
p1 <- ggplot(insurance_data, aes(x = charges)) +
  geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7) +
  labs(title = "Distribution of Insurance Charges", x = "Charges ($)", y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14))

ggsave("plots/01_charges_distribution.png", p1, width = 10, height = 6, dpi = 300)

p2 <- ggplot(insurance_data, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Insurance Charges by Smoking Status", x = "Smoker", y = "Charges ($)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14)) +
  scale_fill_manual(values = c("lightblue", "lightcoral"))

ggsave("plots/02_charges_by_smoking.png", p2, width = 10, height = 6, dpi = 300)

p3 <- ggplot(insurance_data, aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.2) +
  labs(title = "Age vs Charges by Smoking Status", x = "Age (years)", y = "Charges ($)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14)) +
  scale_color_manual(values = c("blue", "red"))

ggsave("plots/03_age_vs_charges.png", p3, width = 10, height = 6, dpi = 300)

cat("Fitting Bayesian models...\n")

# Fit the Bayesian models (simplified for plot generation)
model1 <- stan_glm(charges ~ age + sex + bmi + children + smoker + region,
                   data = insurance_data,
                   family = gaussian(),
                   prior = normal(0, 10000),
                   prior_intercept = normal(0, 10000),
                   chains = 2,  # Reduced for faster computation
                   iter = 1000,
                   seed = 123,
                   refresh = 0)  # Suppress output

model2 <- stan_glm(charges ~ age + sex + bmi + children + smoker + region + 
                   age:smoker + bmi:smoker,
                   data = insurance_data,
                   family = gaussian(),
                   prior = normal(0, 10000),
                   prior_intercept = normal(0, 10000),
                   chains = 2,
                   iter = 1000,
                   seed = 123,
                   refresh = 0)

model3 <- stan_glm(log(charges) ~ age + sex + bmi + children + smoker + region,
                   data = insurance_data,
                   family = gaussian(),
                   prior = normal(0, 10),
                   prior_intercept = normal(0, 10),
                   chains = 2,
                   iter = 1000,
                   seed = 123,
                   refresh = 0)

cat("Generating Bayesian diagnostic plots...\n")

# 2. Posterior Predictive Check
png("plots/04_posterior_predictive_check.png", width = 10, height = 6, units = "in", res = 300)
pp_check(model1, nreps = 50) + 
  labs(title = "Posterior Predictive Check - Model 1") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14))
dev.off()

# 3. MCMC Trace Plot
png("plots/05_mcmc_trace.png", width = 12, height = 8, units = "in", res = 300)
mcmc_trace(model1, pars = c("age", "sexmale", "bmi", "smokeryes")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14))
dev.off()

# 4. Posterior Distributions
png("plots/06_posterior_distributions.png", width = 10, height = 8, units = "in", res = 300)
mcmc_areas(model1, pars = c("age", "sexmale", "bmi", "smokeryes")) +
  labs(title = "Posterior Distributions of Key Parameters") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14))
dev.off()

# 5. Coefficient Plot
png("plots/07_coefficient_plot.png", width = 10, height = 8, units = "in", res = 300)
plot(model1, pars = c("age", "sexmale", "bmi", "children", "smokeryes")) +
  labs(title = "Coefficient Estimates with 95% Credible Intervals") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14))
dev.off()

# 6. Model 2 Interactions
png("plots/08_model2_interactions.png", width = 12, height = 8, units = "in", res = 300)
mcmc_areas(model2, pars = c("age", "sexmale", "bmi", "smokeryes", "age:smokeryes", "bmi:smokeryes")) +
  labs(title = "Model 2: Posterior Distributions with Interaction Terms") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14))
dev.off()

# 7. Residual Analysis
residuals1 <- residuals(model1)
fitted_values1 <- fitted(model1)

png("plots/09_residual_analysis.png", width = 12, height = 8, units = "in", res = 300)
par(mfrow = c(2, 2), cex.main = 1.2)
plot(fitted_values1, residuals1, main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals", pch = 16, col = alpha("blue", 0.6))
abline(h = 0, col = "red", lty = 2, lwd = 2)

qqnorm(residuals1, main = "Normal Q-Q Plot of Residuals", pch = 16, col = alpha("blue", 0.6))
qqline(residuals1, col = "red", lwd = 2)

hist(residuals1, breaks = 30, main = "Distribution of Residuals", 
     xlab = "Residuals", col = "lightblue", border = "white")

plot(fitted_values1, sqrt(abs(residuals1)), main = "Scale-Location Plot",
     xlab = "Fitted Values", ylab = "√|Residuals|", pch = 16, col = alpha("blue", 0.6))
dev.off()

# 8. Model 3 Log-scale Diagnostics
residuals3 <- residuals(model3)
fitted_values3 <- fitted(model3)

png("plots/10_model3_diagnostics.png", width = 12, height = 8, units = "in", res = 300)
par(mfrow = c(2, 2), cex.main = 1.2)
plot(fitted_values3, residuals3, main = "Model 3: Residuals vs Fitted (Log Scale)",
     xlab = "Fitted Log(Charges)", ylab = "Residuals", pch = 16, col = alpha("blue", 0.6))
abline(h = 0, col = "red", lty = 2, lwd = 2)

qqnorm(residuals3, main = "Model 3: Normal Q-Q Plot", pch = 16, col = alpha("blue", 0.6))
qqline(residuals3, col = "red", lwd = 2)

fitted_original <- exp(fitted_values3)
plot(fitted_original, residuals3, main = "Model 3: Residuals vs Fitted (Original Scale)",
     xlab = "Fitted Charges ($)", ylab = "Residuals (Log Scale)", pch = 16, col = alpha("blue", 0.6))
abline(h = 0, col = "red", lty = 2, lwd = 2)

hist(residuals3, breaks = 30, main = "Model 3: Residual Distribution",
     xlab = "Residuals", col = "lightblue", border = "white")
dev.off()

cat("All plots generated successfully in the 'plots' directory!\n")
cat("Generated files:\n")
list.files("plots", pattern = "*.png")