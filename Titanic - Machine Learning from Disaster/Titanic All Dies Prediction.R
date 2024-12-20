setwd("C:/Users/Anirban/Documents/R Studio Projects/Datasets/Titanic - Machine Learning from Disaster")

train <- read.csv(file = "train.csv", stringsAsFactors = FALSE, header = TRUE)
train
View(train)

test <- read.csv(file = "test.csv", stringsAsFactors = FALSE, header = TRUE)
test
View(test)


table(train$Survived)

prop.table(table(train$Survived))

table(train$Sex, train$Survived)


barplot(table(train$Survived), xlab="Survived", ylab="People", main="Train Data Survival")

test$Survived <- rep(0, 418)

prediction <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(prediction, file = "alldies.csv", row.names = FALSE)
