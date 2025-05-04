# Load necessary libraries
install.packages(c("Amelia", "rpart", "rpart.plot", "randomForest", "ggplot2", "caret"))
library(Amelia)  # For missing data visualization
library(rpart)   # For decision tree model
library(rpart.plot)  # For visualizing decision trees
library(randomForest)  # For random forest model
library(ggplot2)  # For additional visualization
library(caret)  # For cross-validation and model evaluation

# Load dataset
train <- read.csv('https://raw.githubusercontent.com/raqueeb/mltraining/master/ML-workbook/train.csv')
test <- read.csv('https://raw.githubusercontent.com/raqueeb/mltraining/master/ML-workbook/test.csv')

# Data Preprocessing

# Handle missing values: Age and Embarked
train[train == ""] <- NA
test[test == ""] <- NA

# Visualize missing data using missmap
missmap(train, main="Titanic Training Data - Missing Map", col=c("yellow", "black"), legend=FALSE)

# Imputation of missing values (using the rpart model)
train$Age[is.na(train$Age)] <- predict(rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked, data=train, method="anova"), 
                                      newdata=train[is.na(train$Age),])
train$Embarked[is.na(train$Embarked)] <- "S"

# Convert categorical variables to factors
train$Survived <- factor(train$Survived, levels = c(0, 1))
train$Pclass <- factor(train$Pclass)
train$Sex <- factor(train$Sex)
train$Embarked <- factor(train$Embarked)
test$Pclass <- factor(test$Pclass)
test$Sex <- factor(test$Sex)
test$Embarked <- factor(test$Embarked)

# Feature Engineering

# Create Family Size variable
train$FamilySize <- train$SibSp + train$Parch + 1
test$FamilySize <- test$SibSp + test$Parch + 1

# Extract Title from Name
train$Title <- sub(".*, (.*)\\..*", "\\1", train$Name)
test$Title <- sub(".*, (.*)\\..*", "\\1", test$Name)

# Missing data visualization and bar plots
# Barplot for Survival
barplot(table(train$Survived), names.arg = c("Perished", "Survived"), main="Survived (Passenger Fate)", col="black")

# Barplot for Passengers' Class
barplot(table(train$Pclass), names.arg = c("First", "Second", "Third"), main="Pclass (Passenger Class)", col="firebrick")

# Barplot for Gender-wise survival
barplot(table(train$Sex, train$Survived), beside=TRUE, col=c("blue", "pink"), 
        legend=c("Did not Survive", "Survived"), main="Survival by Gender")

# Age distribution
hist(train$Age, main="Age Distribution", xlab="Age", col="brown")

# SibSp (Siblings + Spouse) distribution
barplot(table(train$SibSp), main="Siblings + Spouses Aboard", col="darkblue")

# Fare distribution
hist(train$Fare, main="Fare Distribution", xlab="Fare", col="darkgreen")

# Embarked distribution
barplot(table(train$Embarked), names.arg=c("Cherbourg", "Queenstown", "Southampton"), 
        main="Port of Embarkation", col="sienna")

# Modeling

# Train Random Forest Model
rf_model <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                         data=train, importance=TRUE, ntree=100)

# Predict using the Random Forest Model
rf_predictions <- predict(rf_model, test)

# Confusion matrix for model evaluation
confusionMatrix(rf_predictions, test$Survived)

# Decision Tree Model (Alternative model)
dt_model <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                  data=train, method="class")

# Predict using the Decision Tree Model
dt_predictions <- predict(dt_model, test, type="class")

# Confusion matrix for Decision Tree
confusionMatrix(dt_predictions, test$Survived)

# Visualizing the Random Forest Feature Importance
importance(rf_model)
varImpPlot(rf_model, main="Random Forest - Feature Importance")

# Visualizing the Decision Tree Model
rpart.plot(dt_model, extra=106)

# Saving Predictions to CSV
rf_prediction_df <- data.frame(PassengerId = test$PassengerId, Survived = rf_predictions)
write.csv(rf_prediction_df, "Titanic_rf_predictions.csv", row.names = FALSE)

dt_prediction_df <- data.frame(PassengerId = test$PassengerId, Survived = dt_predictions)
write.csv(dt_prediction_df, "Titanic_dt_predictions.csv", row.names = FALSE)

# Model Comparison Plot
model_comparison_df <- data.frame(Model = c("Random Forest", "Decision Tree"), 
                                  Accuracy = c(mean(rf_predictions == test$Survived), mean(dt_predictions == test$Survived)))

ggplot(model_comparison_df, aes(x=Model, y=Accuracy, fill=Model)) +
  geom_bar(stat="identity", show.legend=FALSE) +
  geom_text(aes(label=sprintf("%.2f", Accuracy)), vjust=-0.3) +
  ggtitle("Model Comparison") +
  ylab("Accuracy") +
  theme_minimal()

# Final Output: Random Forest Model Predicted Data saved
print("Random Forest and Decision Tree Predictions are saved to CSV files.")
