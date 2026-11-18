library(tidyverse)
library(car)
library(glmnet)
library(e1071)
library(pROC)
library(caret)
library(corrplot)

set.seed(42)

# load and clean data
df <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", stringsAsFactors = TRUE, fileEncoding = "UTF-8-BOM")
str(df)
summary(df)
sum(is.na(df))
constant_cols <- names(df)[sapply(df, function(x) length(unique(x)) == 1)]
df <- df %>% select(-all_of(constant_cols), -EmployeeNumber)

table(df$Attrition)
prop.table(table(df$Attrition)) * 100

# outlier detection (IQR method)
detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25); q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  sum(x < (q1 - 1.5 * iqr) | x > (q3 + 1.5 * iqr))
}
sapply(df[c("MonthlyIncome", "DistanceFromHome", "YearsAtCompany",
            "TotalWorkingYears", "NumCompaniesWorked")], detect_outliers)

# descriptive summary
df %>%
  group_by(Attrition) %>%
  summarise(across(c(Age, MonthlyIncome, YearsAtCompany, WorkLifeBalance,
                      JobSatisfaction), mean), .groups = "drop")

ggplot(df, aes(x = Department, fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", title = "Attrition Rate by Department") +
  theme_minimal()

ggplot(df, aes(x = factor(WorkLifeBalance), fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(x = "Work-Life Balance Score", y = "Proportion",
       title = "Attrition Rate by Work-Life Balance") +
  theme_minimal()

ggplot(df, aes(x = OverTime, y = WorkLifeBalance, fill = OverTime)) +
  geom_boxplot() +
  labs(title = "Work-Life Balance by Overtime Status") +
  theme_minimal()

# income, age, tenure by attrition
ggplot(df, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_boxplot() + labs(title = "Monthly Income by Attrition") + theme_minimal()

ggplot(df, aes(x = Attrition, y = Age, fill = Attrition)) +
  geom_boxplot() + labs(title = "Age by Attrition") + theme_minimal()

ggplot(df, aes(x = Attrition, y = YearsAtCompany, fill = Attrition)) +
  geom_boxplot() + labs(title = "Years at Company by Attrition") + theme_minimal()

# attrition rate by job role
df %>%
  group_by(JobRole) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100) %>%
  arrange(AttritionRate) %>%
  ggplot(aes(x = reorder(JobRole, AttritionRate), y = AttritionRate)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(x = "Job Role", y = "Attrition Rate (%)", title = "Attrition Rate by Job Role") +
  theme_minimal()

# satisfaction cluster correlation
sat_vars <- df %>% select(JobSatisfaction, EnvironmentSatisfaction,
                           RelationshipSatisfaction, WorkLifeBalance, JobInvolvement)
corrplot(cor(sat_vars), method = "color", addCoef.col = "black", tl.col = "black")

# full numeric correlation matrix
num_vars <- df %>% select(where(is.numeric))
corrplot(cor(num_vars), method = "color", type = "upper", tl.col = "black", tl.cex = 0.6)

# statistical inference
leveneTest(WorkLifeBalance ~ OverTime, data = df)
t.test(WorkLifeBalance ~ OverTime, data = df, var.equal = TRUE)

tab_ot <- table(df$OverTime, df$Attrition)
prop.test(tab_ot[, "Yes"], rowSums(tab_ot))

anova_dept <- aov(JobSatisfaction ~ Department, data = df)
summary(anova_dept)

anova_role <- aov(WorkLifeBalance ~ JobRole, data = df)
summary(anova_role)

leveneTest(MonthlyIncome ~ Department, data = df)

chisq.test(table(df$WorkLifeBalance, df$Attrition))
chisq.test(table(df$OverTime, df$Attrition))

# predictive modelling
train_idx <- createDataPartition(df$Attrition, p = 0.75, list = FALSE)
train <- df[train_idx, ]
test  <- df[-train_idx, ]

log_model <- glm(Attrition ~ ., data = train, family = "binomial")
log_prob <- predict(log_model, newdata = test, type = "response")
log_pred <- factor(ifelse(log_prob > 0.5, "Yes", "No"), levels = levels(df$Attrition))

log_cm <- confusionMatrix(log_pred, test$Attrition, positive = "Yes")
log_auc <- auc(roc(test$Attrition, log_prob))

coef_tab <- summary(log_model)$coefficients
head(coef_tab[order(-abs(coef_tab[, "Estimate"])), ], 10)

x_train <- model.matrix(Attrition ~ . - 1, data = train)
y_train <- train$Attrition
x_test  <- model.matrix(Attrition ~ . - 1, data = test)

lasso_cv <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
lasso_prob <- predict(lasso_cv, newx = x_test, s = "lambda.min", type = "response")
lasso_pred <- factor(ifelse(lasso_prob > 0.5, "Yes", "No"), levels = levels(df$Attrition))
lasso_cm <- confusionMatrix(lasso_pred, test$Attrition, positive = "Yes")
lasso_auc <- auc(roc(test$Attrition, as.numeric(lasso_prob)))
sum(coef(lasso_cv, s = "lambda.min") != 0)

ridge_cv <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)
ridge_prob <- predict(ridge_cv, newx = x_test, s = "lambda.min", type = "response")
ridge_pred <- factor(ifelse(ridge_prob > 0.5, "Yes", "No"), levels = levels(df$Attrition))
ridge_cm <- confusionMatrix(ridge_pred, test$Attrition, positive = "Yes")
ridge_auc <- auc(roc(test$Attrition, as.numeric(ridge_prob)))

nb_model <- naiveBayes(Attrition ~ ., data = train)
nb_pred <- predict(nb_model, newdata = test)
nb_prob  <- predict(nb_model, newdata = test, type = "raw")[, "Yes"]

nb_cm <- confusionMatrix(nb_pred, test$Attrition, positive = "Yes")
nb_auc <- auc(roc(test$Attrition, nb_prob))

# confusion matrix heatmaps
plot_cm <- function(cm, title) {
  as.data.frame(cm$table) %>%
    ggplot(aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 5) +
    scale_fill_gradient(low = "steelblue", high = "darkred") +
    labs(title = title) +
    theme_minimal()
}
plot_cm(log_cm, "Logistic Regression Confusion Matrix")
plot_cm(nb_cm, "Naive Bayes Confusion Matrix")

# ROC comparison plot
roc_log   <- roc(test$Attrition, log_prob)
roc_lasso <- roc(test$Attrition, as.numeric(lasso_prob))
roc_nb    <- roc(test$Attrition, nb_prob)

plot(roc_log, col = "blue", main = "ROC Curve Comparison")
lines(roc_lasso, col = "red")
lines(roc_nb, col = "orange")
legend("bottomright", legend = c("Logistic", "LASSO", "Naive Bayes"),
       col = c("blue", "red", "orange"), lwd = 2)

# model comparison table and chart
model_comparison <- tibble(
  Model = c("Logistic Regression", "LASSO", "Ridge", "Naive Bayes"),
  Accuracy = c(log_cm$overall["Accuracy"], lasso_cm$overall["Accuracy"],
               ridge_cm$overall["Accuracy"], nb_cm$overall["Accuracy"]),
  AUC = c(log_auc, lasso_auc, ridge_auc, nb_auc)
)
model_comparison

model_comparison %>%
  pivot_longer(c(Accuracy, AUC), names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(title = "Model Performance Comparison") +
  theme_minimal()

# PCA on satisfaction variables
pca_model <- prcomp(sat_vars, scale. = TRUE)
summary(pca_model)
pca_model$rotation[, 1:2]
plot(pca_model, type = "l", main = "PCA Scree Plot")
