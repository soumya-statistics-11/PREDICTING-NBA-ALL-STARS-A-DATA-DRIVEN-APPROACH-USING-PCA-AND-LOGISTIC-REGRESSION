rm(list=ls())
setwd("C:\\Users\\User\\Desktop")
nba_data=read.csv("NBA_DATA_(2021-2024).csv")


#EDA------------------------------------------------------
summary(nba_data)
table(nba_data$All_star)  # Count of All-Stars vs. Non-All-Stars

# 1.  Visual representation
library(ggplot2)
ggplot(nba_data, aes(x = as.factor(All_star), fill = as.factor(All_star))) +
  geom_bar() +
  labs(title = "All-Star vs Non-All-Star Distribution", x = "All-Star Status", y = "Count") +
  scale_fill_manual(values = c("red", "blue"))


#2. 
library(ggcorrplot)
cor_matrix <- cor(data[, sapply(data, is.numeric)])  # Compute correlation
ggcorrplot(cor_matrix, hc.order = TRUE, type = "lower", lab = TRUE)

#3
top_scorers <- nba_data[order(-nba_data$PTS),][1:30, c("Player", "Season" , "PTS", "All_star")]
print(top_scorers)

#4. 
ggplot(nba_data, aes(x = as.factor(All_star), y = PTS, fill = as.factor(All_star))) +
  geom_boxplot() +
  labs(title = "Points Distribution by All-Star Status", x = "All-Star", y = "Points")

ggplot(nba_data, aes(x = as.factor(All_star), y = AST, fill = as.factor(All_star))) +
  geom_boxplot() +
  labs(title = "Assists Distribution by All-Star Status", x = "All-Star", y = "Assists")

ggplot(nba_data, aes(x = as.factor(All_star), y = REB, fill = as.factor(All_star))) +
  geom_boxplot() +
  labs(title = "Rebounds Distribution by All-Star Status", x = "All-Star", y = "Rebounds")

#5. ggplot(nba_data, aes(x = PTS, y = AST, color = as.factor(All_star))) +
geom_point(alpha = 0.7) +
  labs(title = "Points vs Assists", x = "Points", y = "Assists") +
  scale_color_manual(values = c("red", "blue"))

#6
ggplot(nba_data, aes(x = as.factor(All_star), y = `Plus_Minus`, fill = as.factor(All_star))) +
  geom_boxplot() +
  labs(title = "Plus/Minus by All-Star Status", x = "All-Star", y = "Plus/Minus")

#7. 
library(ggplot2)

# Filter dataset to include only All-Star players (assuming 1 represents selection)
nba_allstars <- subset(nba_data, All_star == 1)

# Create bar plot
ggplot(nba_allstars, aes(x = Team, fill = as.factor(All_star))) +
  geom_bar() +
  labs(title = "Number of All-Stars Per Team", x = "Team", y = "Count") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))




str(nba_data)  # Check data structure
head(nba_data) # View first few rows
nba_data$All_star <- as.factor(nba_data$All_star)
ggplot(nba_data, aes(x = PTS, y = AST, color = as.factor(All_star))) +
  geom_point(alpha = 0.7) +
  labs(title = "Points vs Assists", x = "Points", y = "Assists") +
  scale_color_manual(values = c("1" = "red", "0" = "blue"))  # Ensure proper mapping



#PCA------------------------------------------------------
library(tidyverse)

# Select only numeric variables for PCA
nba_numeric <- nba_data %>% select(where(is.numeric))
nba_pca_data <- nba_numeric[, !names(nba_numeric) %in% "All_star"]


# Standardize the data
nba_scaled <- scale(nba_pca_data)

# Perform PCA
pca_result <- prcomp(nba_scaled, center = TRUE, scale. = TRUE)

# View PCA summary (variance explained)
summary(pca_result)

# Scree plot to determine number of PCs to retain
screeplot(pca_result, type = "lines", main = "Scree Plot")

# Get principal component scores and add them to the original dataset
nba_pca_final <- bind_cols(nba_data, as.data.frame(pca_result$x))

# View final dataset with PCA scores
head(nba_pca_final)


#Logistic Regression--------------------------------------------------


# Logistic regression using top 3 PCs (adjust as needed)

logit_model <- glm(All_star ~ PC1 + PC2 + PC3+PC4+PC5+PC6, data = nba_pca_final, family = "binomial")

# Summary of logistic regression model
summary(logit_model)
# Get predicted probabilities from the logistic regression model
predicted_probs <- predict(logit_model, type = "response")


# Load pROC library
library(pROC)


# Compute the ROC curve
roc_curve <- roc(nba_pca_final$All_star, predicted_probs)

# Plot the ROC curve
plot(roc_curve, col="blue", main="ROC Curve for NBA All-Star Prediction")
abline(a=0, b=1, lty=2, col="red")  # Add diagonal reference line

# Calculate and print the AUC (Area Under Curve)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))



# Load necessary library
library(caret)

# Generate predictions (Assuming pred_classes contains predicted values: 1 or 0)
pred_classes <- ifelse(predicted_probs > 0.5, 1, 0)  # Convert probabilities to class labels

# Compute Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(nba_numeric$All_star),as.factor(pred_classes))

# Print Confusion Matrix
print(conf_matrix)

TP <- conf_matrix$table[2,2]  # True Positives
FN <- conf_matrix$table[2,1]  # False Negatives
FP <- conf_matrix$table[1,2]  # False Positives
TN <- conf_matrix$table[1,1]  # True Negatives

# Print results
print(paste("TP:", TP, "FN:", FN, "FP:", FP, "TN:", TN))







#Prediction------------------------------------------------------------



library(caret)


# Load new season data
nba_2024 <- read.csv("NBA_2024.csv")

# Remove non-numeric columns (e.g., Player Name, Team)
nba_2024_numeric <- nba_2024[, sapply(nba_2024, is.numeric)]

# Standardize using training data parameters
nba_2024_scaled <- scale(nba_2024_numeric)

colnames(nba_scaled)   # The dataset used for PCA training
colnames(nba_2024_scaled)   # The new dataset for prediction



# Apply PCA transformation
nba_2024_pca <- predict(pca_result, newdata = nba_2024_scaled)

# Select the first few principal components (same as used in logistic regression)
nba_2024_pca_selected <- nba_2024_pca[, 1:6]  # Adjust 'num_pcs' as per Scree Plot

# Predict All-Star probability
nba_2024$All_star_prob <- predict(logit_model, newdata = as.data.frame(nba_2024_pca_selected), type = "response")

# Classify All-Star (Threshold = 0.5)
nba_2024$Predicted_All_star <- ifelse(nba_2024$All_star_prob > 0.5, 1, 0)

# View predicted All-Star players
predicted_all_stars <- nba_2024[nba_2024$Predicted_All_star == 1, c("Player", "All_star_prob")]
print(predicted_all_stars)

# Save predictions
write.csv(predicted_all_stars, "NBA_2024_Predictions_new.csv", row.names = FALSE)
 



















