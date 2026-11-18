library(tidyverse)
library(car)
library(glmnet)
library(e1071)
library(pROC)
library(caret)
library(corrplot)

set.seed(42)
dir.create("plots", showWarnings = FALSE)

# load and clean data
df <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", stringsAsFactors = TRUE, fileEncoding = "UTF-8-BOM")
constant_cols <- names(df)[sapply(df, function(x) length(unique(x)) == 1)]
df <- df %>% select(-all_of(constant_cols), -EmployeeNumber)

# descriptive charts
png("plots/chart_01_attrition_by_department.png", width = 900, height = 600, res = 120)
print(ggplot(df, aes(x = Department, fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", title = "Attrition Rate by Department") +
  theme_minimal())
dev.off()

png("plots/chart_02_attrition_by_wlb.png", width = 900, height = 600, res = 120)
print(ggplot(df, aes(x = factor(WorkLifeBalance), fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(x = "Work-Life Balance Score", y = "Proportion",
       title = "Attrition Rate by Work-Life Balance") +
  theme_minimal())
dev.off()

png("plots/chart_03_wlb_by_overtime.png", width = 900, height = 600, res = 120)
print(ggplot(df, aes(x = OverTime, y = WorkLifeBalance, fill = OverTime)) +
  geom_boxplot() +
  labs(title = "Work-Life Balance by Overtime Status") +
  theme_minimal())
dev.off()

png("plots/chart_04_income_by_attrition.png", width = 800, height = 600, res = 120)
print(ggplot(df, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_boxplot() + labs(title = "Monthly Income by Attrition") + theme_minimal())
dev.off()

png("plots/chart_05_age_by_attrition.png", width = 800, height = 600, res = 120)
print(ggplot(df, aes(x = Attrition, y = Age, fill = Attrition)) +
  geom_boxplot() + labs(title = "Age by Attrition") + theme_minimal())
dev.off()

png("plots/chart_06_tenure_by_attrition.png", width = 800, height = 600, res = 120)
print(ggplot(df, aes(x = Attrition, y = YearsAtCompany, fill = Attrition)) +
  geom_boxplot() + labs(title = "Years at Company by Attrition") + theme_minimal())
dev.off()

png("plots/chart_07_attrition_by_jobrole.png", width = 900, height = 650, res = 120)
print(df %>%
  group_by(JobRole) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100) %>%
  arrange(AttritionRate) %>%
  ggplot(aes(x = reorder(JobRole, AttritionRate), y = AttritionRate)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(x = "Job Role", y = "Attrition Rate (%)", title = "Attrition Rate by Job Role") +
  theme_minimal())
dev.off()

# correlation charts
sat_vars <- df %>% select(JobSatisfaction, EnvironmentSatisfaction,
                           RelationshipSatisfaction, WorkLifeBalance, JobInvolvement)
png("plots/chart_08_satisfaction_correlation.png", width = 800, height = 700, res = 120)
corrplot(cor(sat_vars), method = "color", addCoef.col = "black", tl.col = "black")
dev.off()

num_vars <- df %>% select(where(is.numeric))
png("plots/chart_09_full_correlation_matrix.png", width = 1100, height = 1000, res = 120)
corrplot(cor(num_vars), method = "color", type = "upper", tl.col = "black", tl.cex = 0.6)
dev.off()

# predictive models
train_idx <- createDataPartition(df$Attrition, p = 0.75, list = FALSE)
train <- df[train_idx, ]
test  <- df[-train_idx, ]

log_model <- glm(Attrition ~ ., data = train, family = "binomial")
log_prob <- predict(log_model, newdata = test, type = "response")
log_pred <- factor(ifelse(log_prob > 0.5, "Yes", "No"), levels = levels(df$Attrition))
log_cm <- confusionMatrix(log_pred, test$Attrition, positive = "Yes")

x_train <- model.matrix(Attrition ~ . - 1, data = train)
y_train <- train$Attrition
x_test  <- model.matrix(Attrition ~ . - 1, data = test)

lasso_cv <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
lasso_prob <- predict(lasso_cv, newx = x_test, s = "lambda.min", type = "response")

nb_model <- naiveBayes(Attrition ~ ., data = train)
nb_prob  <- predict(nb_model, newdata = test, type = "raw")[, "Yes"]
nb_pred  <- predict(nb_model, newdata = test)
nb_cm    <- confusionMatrix(nb_pred, test$Attrition, positive = "Yes")

# confusion matrix heatmaps
plot_cm <- function(cm, title, fname) {
  p <- as.data.frame(cm$table) %>%
    ggplot(aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 5) +
    scale_fill_gradient(low = "steelblue", high = "darkred") +
    labs(title = title) +
    theme_minimal()
  png(file.path("plots", fname), width = 700, height = 600, res = 120)
  print(p)
  dev.off()
}
plot_cm(log_cm, "Logistic Regression Confusion Matrix", "chart_10_cm_logistic.png")
plot_cm(nb_cm, "Naive Bayes Confusion Matrix", "chart_11_cm_naivebayes.png")

# ROC comparison
roc_log   <- roc(test$Attrition, log_prob, quiet = TRUE)
roc_lasso <- roc(test$Attrition, as.numeric(lasso_prob), quiet = TRUE)
roc_nb    <- roc(test$Attrition, nb_prob, quiet = TRUE)

png("plots/chart_12_roc_comparison.png", width = 800, height = 700, res = 120)
plot(roc_log, col = "blue", main = "ROC Curve Comparison")
lines(roc_lasso, col = "red")
lines(roc_nb, col = "orange")
legend("bottomright", legend = c("Logistic", "LASSO", "Naive Bayes"),
       col = c("blue", "red", "orange"), lwd = 2)
dev.off()

# model comparison chart
model_comparison <- tibble(
  Model = c("Logistic Regression", "LASSO", "Naive Bayes"),
  Accuracy = c(log_cm$overall["Accuracy"], mean((ifelse(lasso_prob > 0.5, "Yes", "No")) == test$Attrition), nb_cm$overall["Accuracy"]),
  AUC = c(auc(roc_log), auc(roc_lasso), auc(roc_nb))
)
png("plots/chart_13_model_comparison.png", width = 900, height = 600, res = 120)
print(model_comparison %>%
  pivot_longer(c(Accuracy, AUC), names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(title = "Model Performance Comparison") +
  theme_minimal())
dev.off()

# PCA scree plot
pca_model <- prcomp(sat_vars, scale. = TRUE)
png("plots/chart_14_pca_scree.png", width = 800, height = 600, res = 120)
plot(pca_model, type = "l", main = "PCA Scree Plot")
dev.off()

cat("All charts saved to the plots/ folder\n")
