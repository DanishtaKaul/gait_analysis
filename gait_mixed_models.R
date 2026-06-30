library(readr)
library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(effectsize)
library(performance)
library(see)
library(car)
library(nlme)

# ==========================================================
# GAIT MIXED MODELS
# Dataset: pooled gait features (one row per participant x condition)
# ==========================================================

# Load pooled gait dataset
base_dir <- "D:/Gait_Analysis"
gait_df <- read_csv(file.path(base_dir, "rf_pooled_gait_features.csv"))

# ==========================================================
# Remove PID 49
# ==========================================================

gait_df <- gait_df[gait_df$pid != "PID 49", ]

# Check dataset structure
str(gait_df)
names(gait_df)

# ==========================================================
# Convert grouping variables to factors
# ==========================================================

gait_df$pid <- factor(gait_df$pid)
gait_df$age <- factor(gait_df$age)
gait_df$light <- factor(gait_df$light)
gait_df$obstacle <- factor(gait_df$obstacle)

str(gait_df)

# ==========================================================
# Set reference levels for factors
# ==========================================================

gait_df$age <- relevel(gait_df$age, ref = "young")
gait_df$light <- relevel(gait_df$light, ref = "light")
gait_df$obstacle <- relevel(gait_df$obstacle, ref = "expected_absent")

levels(gait_df$age)
levels(gait_df$light)
levels(gait_df$obstacle)

# ==========================================================
# SPEED MODEL
# ==========================================================

m_speed <- lmer(speed_mean ~ age * light + age * obstacle + (1|pid), data = gait_df)

resid_speed <- residuals(m_speed)
fitted_speed <- fitted(m_speed)

summary(m_speed)
anova(m_speed)

eta_squared(m_speed, partial = TRUE)

# Check model assumptions
dev.new(); check_model(m_speed, check="pp_check")
dev.new(); check_model(m_speed, check="linearity")
dev.new(); check_model(m_speed, check="homogeneity")
dev.new(); check_model(m_speed, check="outliers")
dev.new(); check_model(m_speed, check="vif")
dev.new(); check_model(m_speed, check="normality")
dev.new(); check_model(m_speed, check="reqq")

emmeans(m_speed, pairwise ~ light, adjust = "bonferroni")

emm_obs <- emmeans(m_speed, ~ obstacle)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")

#emmeans(m_speed, ~ obstacle)
emmeans(m_speed, ~ age)

# ==========================================================
# STRIDE LENGTH MODEL
# ==========================================================

m_stride_length <- lmer(stride_length_mean ~ age * light + age * obstacle + (1|pid), data = gait_df)

resid_stride_length <- residuals(m_stride_length)
fitted_stride_length <- fitted(m_stride_length)

summary(m_stride_length)
anova(m_stride_length)

eta_squared(m_stride_length, partial = TRUE)

dev.new(); check_model(m_stride_length, check="pp_check")
dev.new(); check_model(m_stride_length, check="linearity")
dev.new(); check_model(m_stride_length, check="homogeneity")
dev.new(); check_model(m_stride_length, check="outliers")
dev.new(); check_model(m_stride_length, check="vif")
dev.new(); check_model(m_stride_length, check="normality")
dev.new(); check_model(m_stride_length, check="reqq")

#unequal variance model
m_sl_nlme <- lme(stride_length_mean ~ age * light + age * obstacle, 
                 random = ~1|pid, data = gait_df, method = "REML")

#unequal variance by age
m_sl_varAge <- lme(stride_length_mean ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML",
                   weights = varIdent(form = ~1|age))

#unequal variance by light
m_sl_varLight <- lme(stride_length_mean ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|light))

#unequal variance by obstacle
m_sl_varObs <- lme(stride_length_mean ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML",
                   weights = varIdent(form = ~1|obstacle))

anova(m_sl_nlme, m_sl_varAge)
anova(m_sl_nlme, m_sl_varLight)
anova(m_sl_nlme, m_sl_varObs)


gait_df$light_obs <- interaction(gait_df$light, gait_df$obstacle)

m_sl_varLightObs <- lme(stride_length_mean ~ age * light + age * obstacle, 
                        random = ~1|pid, data = gait_df, method = "REML",
                        weights = varIdent(form = ~1|light_obs))
#combined model not sig
anova(m_sl_nlme, m_sl_varLightObs)

summary(m_sl_varLight)
anova(m_sl_varLight)


# partial eta square
anova_sl <- anova(m_sl_varLight)
f_vals <- anova_sl$`F-value`[2:6]
num_df <- anova_sl$numDF[2:6]
den_df <- anova_sl$denDF[2:6]
eta2p <- (f_vals * num_df) / (f_vals * num_df + den_df)
names(eta2p) <- c("age", "light", "obstacle", "age:light", "age:obstacle")
round(eta2p, 3)

# Emmeans
emmeans(m_sl_varLight, pairwise ~ light | age, adjust = "bonferroni")
emm_obs <- emmeans(m_sl_varLight, ~ obstacle)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")


# ==========================================================
# STRIDE TIME MODEL
# ==========================================================

m_stride_time <- lmer(stride_time_mean ~ age * light + age * obstacle + (1|pid), data = gait_df)

resid_stride_time <- residuals(m_stride_time)
fitted_stride_time <- fitted(m_stride_time)

summary(m_stride_time)
anova(m_stride_time)

eta_squared(m_stride_time, partial = TRUE)

dev.new(); check_model(m_stride_time, check="pp_check")
dev.new(); check_model(m_stride_time, check="linearity")
dev.new(); check_model(m_stride_time, check="homogeneity")
dev.new(); check_model(m_stride_time, check="outliers")
dev.new(); check_model(m_stride_time, check="vif")
dev.new(); check_model(m_stride_time, check="normality")
dev.new(); check_model(m_stride_time, check="reqq")

idx <- which.max(abs(residuals(m_stride_time)))
gait_df[idx, ]

#unequal variance model
m_st_nlme <- lme(stride_time_mean ~ age * light + age * obstacle, 
                 random = ~1|pid, data = gait_df, method = "REML")

#unequal variance by age
m_st_varAge <- lme(stride_time_mean ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML",
                   weights = varIdent(form = ~1|age))

#unequal variance by light
m_st_varLight <- lme(stride_time_mean ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|light))

#unequal variance by obstacle
m_st_varObs <- lme(stride_time_mean ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML",
                   weights = varIdent(form = ~1|obstacle))

anova(m_st_nlme, m_st_varAge)
anova(m_st_nlme, m_st_varLight)
anova(m_st_nlme, m_st_varObs)

#light obstacle combined model
m_st_varLightObs <- lme(stride_time_mean ~ age * light + age * obstacle, 
                        random = ~1|pid, data = gait_df, method = "REML",
                        weights = varIdent(form = ~1|light_obs))

anova(m_st_nlme, m_st_varLightObs)

#parsimony test
anova(m_st_varObs, m_st_varLightObs)

#combined model is sig

summary(m_st_varLightObs)
anova(m_st_varLightObs)

# partial eta square
anova_st <- anova(m_st_varLightObs)
f_vals <- anova_st$`F-value`[2:6]
num_df <- anova_st$numDF[2:6]
den_df <- anova_st$denDF[2:6]
eta2p <- (f_vals * num_df) / (f_vals * num_df + den_df)
names(eta2p) <- c("age", "light", "obstacle", "age:light", "age:obstacle")
round(eta2p, 3)

# Emmeans
emmeans(m_st_varLightObs, pairwise ~ light | age, adjust = "bonferroni")

emm_obs <- emmeans(m_st_varLightObs, ~ obstacle)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")

# ==========================================================
# CADENCE MODEL
# ==========================================================

m_cadence <- lmer(cadence ~ age * light + age * obstacle + (1|pid), data = gait_df)

resid_cadence <- residuals(m_cadence)
fitted_cadence <- fitted(m_cadence)

summary(m_cadence)
anova(m_cadence)

eta_squared(m_cadence, partial = TRUE)

dev.new(); check_model(m_cadence, check="pp_check")
dev.new(); check_model(m_cadence, check="linearity")
dev.new(); check_model(m_cadence, check="homogeneity")
dev.new(); check_model(m_cadence, check="outliers")
dev.new(); check_model(m_cadence, check="vif")
dev.new(); check_model(m_cadence, check="normality")
dev.new(); check_model(m_cadence, check="reqq")

#unequal variance model
m_cad_nlme <- lme(cadence ~ age * light + age * obstacle, 
                  random = ~1|pid, data = gait_df, method = "REML")

#unequal variance by age
m_cad_varAge <- lme(cadence ~ age * light + age * obstacle, 
                    random = ~1|pid, data = gait_df, method = "REML",
                    weights = varIdent(form = ~1|age))

#unequal variance by light
m_cad_varLight <- lme(cadence ~ age * light + age * obstacle, 
                      random = ~1|pid, data = gait_df, method = "REML",
                      weights = varIdent(form = ~1|light))

#unequal variance by obstacle
m_cad_varObs <- lme(cadence ~ age * light + age * obstacle, 
                    random = ~1|pid, data = gait_df, method = "REML",
                    weights = varIdent(form = ~1|obstacle))

anova(m_cad_nlme, m_cad_varAge)
anova(m_cad_nlme, m_cad_varLight)
anova(m_cad_nlme, m_cad_varObs)

#combined light and obstacle model
m_cad_varLightObs <- lme(cadence ~ age * light + age * obstacle, 
                         random = ~1|pid, data = gait_df, method = "REML",
                         weights = varIdent(form = ~1|light_obs))

anova(m_cad_nlme, m_cad_varLightObs)

#parsimony test
anova(m_cad_varLight, m_cad_varLightObs)

#combined model significant

summary(m_cad_varLightObs)
anova(m_cad_varLightObs)

# partial eta square
anova_cad <- anova(m_cad_varLightObs)
f_vals <- anova_cad$`F-value`[2:6]
num_df <- anova_cad$numDF[2:6]
den_df <- anova_cad$denDF[2:6]
eta2p <- (f_vals * num_df) / (f_vals * num_df + den_df)
names(eta2p) <- c("age", "light", "obstacle", "age:light", "age:obstacle")
round(eta2p, 3)

# Emmeans - all collapsed, no significant interactions
emmeans(m_cad_varLightObs, pairwise ~ light, adjust = "bonferroni")

emm_obs <- emmeans(m_cad_varLightObs, ~ obstacle)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")

emmeans(m_cad_varLightObs, ~ age)


# ==========================================================
# STEP WIDTH MODEL
# ==========================================================

m_step_width <- lmer(step_width_mean ~ age * light + age * obstacle + (1|pid), data = gait_df)

resid_step_width <- residuals(m_step_width)
fitted_step_width <- fitted(m_step_width)

summary(m_step_width)
anova(m_step_width)

eta_squared(m_step_width, partial = TRUE)

dev.new(); check_model(m_step_width, check="pp_check")
dev.new(); check_model(m_step_width, check="linearity")
dev.new(); check_model(m_step_width, check="homogeneity")
dev.new(); check_model(m_step_width, check="outliers")
dev.new(); check_model(m_step_width, check="vif")
dev.new(); check_model(m_step_width, check="normality")
dev.new(); check_model(m_step_width, check="reqq")

# Base nlme model (no variance structure)
m_sw_nlme <- lme(step_width_mean ~ age * light + age * obstacle, 
                 random = ~1|pid, data = gait_df, method = "REML")

# Unequal variance by age
m_sw_varAge <- lme(step_width_mean ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML",
                   weights = varIdent(form = ~1|age))

# Unequal variance by light
m_sw_varLight <- lme(step_width_mean ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|light))

# Unequal variance by obstacle
m_sw_varObs <- lme(step_width_mean ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML",
                   weights = varIdent(form = ~1|obstacle))

anova(m_sw_nlme, m_sw_varAge)
anova(m_sw_nlme, m_sw_varLight)
anova(m_sw_nlme, m_sw_varObs)

# Combined light × obstacle (only if individual ones are significant)
m_sw_varLightObs <- lme(step_width_mean ~ age * light + age * obstacle, 
                        random = ~1|pid, data = gait_df, method = "REML",
                        weights = varIdent(form = ~1|light_obs))
anova(m_sw_nlme, m_sw_varLightObs)

#parsimony test
anova(m_sw_varLight, m_sw_varLightObs)

# Final model: varLightObs (combined vs base significant)
summary(m_sw_varLightObs)
anova(m_sw_varLightObs)

# partial eta square
anova_sw <- anova(m_sw_varLightObs)
f_vals <- anova_sw$`F-value`[2:6]
num_df <- anova_sw$numDF[2:6]
den_df <- anova_sw$denDF[2:6]
eta2p <- (f_vals * num_df) / (f_vals * num_df + den_df)
names(eta2p) <- c("age", "light", "obstacle", "age:light", "age:obstacle")
round(eta2p, 3)

# Emmeans
emmeans(m_sw_varLightObs, pairwise ~ light | age, adjust = "bonferroni")

emm_obs <- emmeans(m_sw_varLightObs, ~ obstacle)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")


# Exploratory all-pairwise (planned contrasts non-significant) only ua vs up sig
emmeans(m_sw_varLightObs, pairwise ~ obstacle, adjust = "bonferroni")
# ==========================================================
# LATERAL DEVIATION MODEL
# ==========================================================
m_lateral_dev <- lmer(lateral_path_dev ~ age * light + age * obstacle + (1|pid), data = gait_df)

resid_lateral_dev <- residuals(m_lateral_dev)
fitted_lateral_dev <- fitted(m_lateral_dev)

summary(m_lateral_dev)
anova(m_lateral_dev)

eta_squared(m_lateral_dev, partial = TRUE)

dev.new(); check_model(m_lateral_dev, check="pp_check")
dev.new(); check_model(m_lateral_dev, check="linearity")
dev.new(); check_model(m_lateral_dev, check="homogeneity")
dev.new(); check_model(m_lateral_dev, check="outliers")
dev.new(); check_model(m_lateral_dev, check="vif")
dev.new(); check_model(m_lateral_dev, check="normality")
dev.new(); check_model(m_lateral_dev, check="reqq")


# Base model
m_lat_nlme <- lme(lateral_path_dev ~ age * light + age * obstacle, 
                  random = ~1|pid, data = gait_df, method = "REML")

# Model allowing different residual variance by age
m_lat_varAge <- lme(lateral_path_dev ~ age * light + age * obstacle, 
                    random = ~1|pid, data = gait_df, method = "REML",
                    weights = varIdent(form = ~1|age))

anova(m_lat_nlme, m_lat_varAge)

#diff residual variance by light
m_lat_varLight <- lme(lateral_path_dev ~ age * light + age * obstacle, 
                      random = ~1|pid, data = gait_df, method = "REML",
                      weights = varIdent(form = ~1|light))

#diff residual variance by obstacle
m_lat_varObs <- lme(lateral_path_dev ~ age * light + age * obstacle, 
                    random = ~1|pid, data = gait_df, method = "REML",
                    weights = varIdent(form = ~1|obstacle))

anova(m_lat_nlme, m_lat_varLight)
anova(m_lat_nlme, m_lat_varObs)

# Use varLight model as final
summary(m_lat_varLight)
anova(m_lat_varLight)

# Extract F-values and df from anova output
anova_lat <- anova(m_lat_varLight)

# Skip intercept (row 1), effects are rows 2-6
f_vals <- anova_lat$`F-value`[2:6]
num_df <- anova_lat$numDF[2:6]
den_df <- anova_lat$denDF[2:6]

eta2p <- (f_vals * num_df) / (f_vals * num_df + den_df)

names(eta2p) <- c("age", "light", "obstacle", "age:light", "age:obstacle")
round(eta2p, 3)

emmeans(m_lat_varLight, pairwise ~ light | age, adjust = "bonferroni")

emm_obs <- emmeans(m_lat_varLight, ~ obstacle | age)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")

# ==========================================================
# STRIDE LENGTH CV MODEL
# ==========================================================

m_stride_length_cv <- lmer(stride_length_cv_pct ~ age * light + age * obstacle + (1|pid), data = gait_df)

summary(m_stride_length_cv)
anova(m_stride_length_cv)

dev.new(); check_model(m_stride_length_cv, check="pp_check")
dev.new(); check_model(m_stride_length_cv, check="linearity")
dev.new(); check_model(m_stride_length_cv, check="homogeneity")
dev.new(); check_model(m_stride_length_cv, check="outliers")
dev.new(); check_model(m_stride_length_cv, check="vif")
dev.new(); check_model(m_stride_length_cv, check="normality")
dev.new(); check_model(m_stride_length_cv, check="reqq")

#non equal variance model
m_slcv_nlme <- lme(stride_length_cv_pct ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML")

#non equal variance by age
m_slcv_varAge <- lme(stride_length_cv_pct ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|age))

#non equal variance by light
m_slcv_varLight <- lme(stride_length_cv_pct ~ age * light + age * obstacle, 
                       random = ~1|pid, data = gait_df, method = "REML",
                       weights = varIdent(form = ~1|light))

#non equal variance by obstacle
m_slcv_varObs <- lme(stride_length_cv_pct ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|obstacle))

anova(m_slcv_nlme, m_slcv_varAge)
anova(m_slcv_nlme, m_slcv_varLight)
anova(m_slcv_nlme, m_slcv_varObs)

gait_df$age_obs <- interaction(gait_df$age, gait_df$obstacle)

m_slcv_varAgeObs <- lme(stride_length_cv_pct ~ age * light + age * obstacle, 
                        random = ~1|pid, data = gait_df, method = "REML",
                        weights = varIdent(form = ~1|age_obs))

anova(m_slcv_nlme, m_slcv_varAgeObs)

#parsimony check
anova(m_slcv_varAge, m_slcv_varAgeObs)

#final
summary(m_slcv_varAgeObs)
anova(m_slcv_varAgeObs)

# partial eta square
anova_slcv <- anova(m_slcv_varAgeObs)
f_vals <- anova_slcv$`F-value`[2:6]
num_df <- anova_slcv$numDF[2:6]
den_df <- anova_slcv$denDF[2:6]
eta2p <- (f_vals * num_df) / (f_vals * num_df + den_df)
names(eta2p) <- c("age", "light", "obstacle", "age:light", "age:obstacle")
round(eta2p, 3)

# Emmeans
emmeans(m_slcv_varAgeObs, pairwise ~ light, adjust = "bonferroni")

emm_obs <- emmeans(m_slcv_varAgeObs, ~ obstacle | age)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")

# ==========================================================
# STRIDE TIME CV MODEL
# ==========================================================

m_stride_time_cv <- lmer(stride_time_cv_pct ~ age * light + age * obstacle + (1|pid), data = gait_df)

summary(m_stride_time_cv)
anova(m_stride_time_cv)

dev.new(); check_model(m_stride_time_cv, check="pp_check")
dev.new(); check_model(m_stride_time_cv, check="linearity")
dev.new(); check_model(m_stride_time_cv, check="homogeneity")
dev.new(); check_model(m_stride_time_cv, check="outliers")
dev.new(); check_model(m_stride_time_cv, check="vif")
dev.new(); check_model(m_stride_time_cv, check="normality")
dev.new(); check_model(m_stride_time_cv, check="reqq")

#unequal variance
m_stcv_nlme <- lme(stride_time_cv_pct ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML")

#unequal variance by age
m_stcv_varAge <- lme(stride_time_cv_pct ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|age))

#unequal variance by light
m_stcv_varLight <- lme(stride_time_cv_pct ~ age * light + age * obstacle, 
                       random = ~1|pid, data = gait_df, method = "REML",
                       weights = varIdent(form = ~1|light))

#unequal variance by obstacle
m_stcv_varObs <- lme(stride_time_cv_pct ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|obstacle))

anova(m_stcv_nlme, m_stcv_varAge)
anova(m_stcv_nlme, m_stcv_varLight)
anova(m_stcv_nlme, m_stcv_varObs)

#obstacle variance model
summary(m_stcv_varObs)
anova(m_stcv_varObs)

#effect size
anova_stcv <- anova(m_stcv_varObs)
f_vals <- anova_stcv$`F-value`[2:6]
num_df <- anova_stcv$numDF[2:6]
den_df <- anova_stcv$denDF[2:6]
eta2p <- (f_vals * num_df) / (f_vals * num_df + den_df)
names(eta2p) <- c("age", "light", "obstacle", "age:light", "age:obstacle")
round(eta2p, 3)

emmeans(m_stcv_varObs, pairwise ~ light, adjust = "bonferroni")

emm_obs <- emmeans(m_stcv_varObs, ~ obstacle | age)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")

#exploratory obstacle pairwise
emmeans(m_stcv_varObs, pairwise ~ obstacle | age, adjust = "bonferroni")
# ==========================================================
# STEP WIDTH SD MODEL
# ==========================================================

m_step_width_sd <- lmer(step_width_sd ~ age * light + age * obstacle + (1|pid), data = gait_df)

summary(m_step_width_sd)
anova(m_step_width_sd)

dev.new(); check_model(m_step_width_sd, check="pp_check")
dev.new(); check_model(m_step_width_sd, check="linearity")
dev.new(); check_model(m_step_width_sd, check="homogeneity")
dev.new(); check_model(m_step_width_sd, check="outliers")
dev.new(); check_model(m_step_width_sd, check="vif")
dev.new(); check_model(m_step_width_sd, check="normality")
dev.new(); check_model(m_step_width_sd, check="reqq")

#unequal variance model
m_swsd_nlme <- lme(step_width_sd ~ age * light + age * obstacle, 
                   random = ~1|pid, data = gait_df, method = "REML")

#unequal variance by age
m_swsd_varAge <- lme(step_width_sd ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|age))

#unequal variance by light
m_swsd_varLight <- lme(step_width_sd ~ age * light + age * obstacle, 
                       random = ~1|pid, data = gait_df, method = "REML",
                       weights = varIdent(form = ~1|light))

#unequal variance by obstacle
m_swsd_varObs <- lme(step_width_sd ~ age * light + age * obstacle, 
                     random = ~1|pid, data = gait_df, method = "REML",
                     weights = varIdent(form = ~1|obstacle))

anova(m_swsd_nlme, m_swsd_varAge)
anova(m_swsd_nlme, m_swsd_varLight)
anova(m_swsd_nlme, m_swsd_varObs)

#use orig lmer model
summary(m_step_width_sd)
anova(m_step_width_sd)

#effect size
eta_squared(m_step_width_sd, partial = TRUE)

emmeans(m_step_width_sd, pairwise ~ light | age, adjust = "bonferroni")

emm_obs <- emmeans(m_step_width_sd, ~ obstacle)
contrast(emm_obs, list(
  "UP - EP" = c(0, -1, 0, 1),
  "UA - EA" = c(-1, 0, 1, 0)
), adjust = "bonferroni")

#exploratory obstacle pairwise
emmeans(m_step_width_sd, pairwise ~ obstacle, adjust = "bonferroni")
