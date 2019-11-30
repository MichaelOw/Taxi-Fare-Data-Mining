suppressMessages(library(dplyr))
suppressMessages(library(data.table))
suppressMessages(library(lubridate))
suppressMessages(library(Matrix))
suppressMessages(library(xgboost))
suppressMessages(library(ggplot2))
suppressMessages(library(caret))

setwd("C:/path/to/your/files")

data_train = fread("train_data.csv")
data_test = fread("test.csv")

#######################
# FEATURE ENGINEERING #
#######################

# Misc Covariates
data_train <- data_train %>% mutate(STRAIGHT_DIST = sqrt( (X_END - X_START)**2 + (Y_END - Y_START)**2))
data_train <- data_train %>% mutate(MANHATTAN_DIST = sqrt( abs(X_END - X_START) + abs(Y_END - Y_START)))
data_train <- data_train %>% mutate(LOG_STRAIGHT_DIST = log(STRAIGHT_DIST))
data_train <- data_train %>% mutate(LOG_MANHATTAN_DIST = log(MANHATTAN_DIST))
data_train <- data_train %>% mutate(LOG_DURATION = log(DURATION))
data_train <- data_train %>% mutate(LOG_TRAJ_LENGTH = log(TRAJ_LENGTH))
data_train <- data_train %>% mutate(HOUR_N = hour(TIMESTAMP))
data_train <- data_train %>% mutate(DAY_OF_WEEK_N = wday(TIMESTAMP))
data_train <- data_train %>% mutate(RUSH_HOUR = (hour(TIMESTAMP) >= 7 & hour(TIMESTAMP) <= 19))
data_train <- data_train %>% mutate(WEEKEND = ((wday(TIMESTAMP) == 1) | (wday(TIMESTAMP) == 7)))

data_test <- data_test %>% mutate(STRAIGHT_DIST = sqrt( (X_END - X_START)**2 + (Y_END - Y_START)**2))
data_test <- data_test %>% mutate(MANHATTAN_DIST = sqrt( abs(X_END - X_START) + abs(Y_END - Y_START)))
data_test <- data_test %>% mutate(LOG_STRAIGHT_DIST = log(STRAIGHT_DIST))
data_test <- data_test %>% mutate(LOG_MANHATTAN_DIST = log(MANHATTAN_DIST))
data_test <- data_test %>% mutate(HOUR = as.factor(hour(TIMESTAMP)))
data_test <- data_test %>% mutate(DAY_OF_WEEK = as.factor(wday(TIMESTAMP)))
data_test <- data_test %>% mutate(HOUR_N = hour(TIMESTAMP))
data_test <- data_test %>% mutate(DAY_OF_WEEK_N = wday(TIMESTAMP))
data_test <- data_test %>% mutate(RUSH_HOUR = (hour(TIMESTAMP) >= 7 & hour(TIMESTAMP) <= 19))
data_test <- data_test %>% mutate(WEEKEND = ((wday(TIMESTAMP) == 1) | (wday(TIMESTAMP) == 7)))

# Trajectory Change Count
TRAJ_COUNT <- unlist(lapply(data_train[,'X_TRAJECTORY'], function(x) length(unlist(strsplit(x, ",")))))
TRAJ_COUNT_LEVEL <- cut(TRAJ_COUNT, 2, include.lowest=TRUE, labels=c(0, 1))
TRAJ_COUNT_LEVEL <- as.numeric(TRAJ_COUNT_LEVEL)
TRAJ_COUNT_LEVEL <- TRAJ_COUNT_LEVEL - 1
data_train <- data.frame(data_train, TRAJ_COUNT_LEVEL)

# Driver Longcut levels - Median
longcut_levels <- data_train %>%
    group_by(TAXI_ID) %>%
    summarize(LONGCUT_LEVEL = median(TRAJ_LENGTH/STRAIGHT_DIST))
longcut_levels <- data.frame(longcut_levels)
data_train <- data_train %>% mutate(LOG_DRIVER_LONGCUT = log(longcut_levels[TAXI_ID,2]))
data_test <- data_test %>% mutate(LOG_DRIVER_LONGCUT = log(longcut_levels[TAXI_ID,2]))

# Driver Average Speed Inv (DURATION/TRAJ_LENGTH)
average_speed_inv <- data_train %>%
    group_by(TAXI_ID) %>%
    summarize(AVERAGE_SPEED_INV = mean(DURATION/TRAJ_LENGTH))
average_speed_inv <- data.frame(average_speed_inv)
data_train <- data_train %>% mutate(LOG_DRIVER_SPEED_INV = log(average_speed_inv[TAXI_ID,2]))
data_test <- data_test %>% mutate(LOG_DRIVER_SPEED_INV = log(average_speed_inv[TAXI_ID,2]))

# Bearing
bearing = function(X_START, Y_START, X_END, Y_END) {
    bearing_angle = atan2( X_END - X_START, Y_END - Y_START ) * (180/pi)
    bearing_angle = case_when(
                        bearing_angle >= 0 ~ bearing_angle,
                        bearing_angle < 0 ~ bearing_angle + 360
                    )
    return(bearing_angle)
}
data_train <- data_train %>% mutate(BEARING = bearing(X_START, Y_START, X_END, Y_END))
data_test <- data_test %>% mutate(BEARING = bearing(X_START, Y_START, X_END, Y_END))

# Remove Unnecessary Columns
data_train <- subset(data_train, select = -c(ID, TIMESTAMP, DURATION, TRAJ_LENGTH, STRAIGHT_DIST, MANHATTAN_DIST, X_TRAJECTORY, Y_TRAJECTORY))
data_test <- subset(data_test, select = -c(ID, TIMESTAMP, STRAIGHT_DIST, MANHATTAN_DIST))





############################
# PREDICT TRAJ_COUNT_LEVEL #
############################

formula <- ~ LOG_STRAIGHT_DIST + LOG_MANHATTAN_DIST + LOG_DRIVER_LONGCUT +
              X_START + Y_START + X_END + Y_END + HOUR_N + WEEKEND + BEARING

x <- sparse.model.matrix(formula, data_train)[,-1]
y <- data_train$TRAJ_COUNT_LEVEL

# Training Grid
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3)
xgb.grid <- expand.grid(nrounds = 2000,
                        max_depth = c(10, 16, 20),
                        eta = c(0.01, 0.1),
                        gamma = 0.0,
                        colsample_bytree = 0.5,
                        subsample = 1.0,
                        min_child_weight = c(1, 25))
xgb_tune1 <- train(x, y, method="xgbTree", metric = "Accuracy", trControl=cv.ctrl, tuneGrid=xgb.grid)
xgb_tune1$bestTune

dtrain <- xgb.DMatrix(data = x, label = y)
params <- list( booster = "gbtree", 
                objective = "binary:logistic", 
                eta=0.1, 
                colsample_bytree=0.5, 
                min_child_weight=25, 
                max_depth=18)
xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                nrounds = 10000, 
                nfold = 5, 
                showsd = T, 
                stratified = T, 
                print.every.n = 10, 
                early.stop.round = 20, 
                maximize = F)
xgbfit <- xgboost(params = params, data = dtrain, nrounds =xgbcv$best_iteration)

x <- sparse.model.matrix(formula, data_test)[,-1]
TRAJ_COUNT_LEVEL_PRED <- predict(xgbfit, newdata = x)
TRAJ_COUNT_LEVEL_PRED <- cut(TRAJ_COUNT_LEVEL_PRED, 2)
TRAJ_COUNT_LEVEL_PRED <- as.numeric(TRAJ_COUNT_LEVEL_PRED)
TRAJ_COUNT_LEVEL_PRED <- TRAJ_COUNT_LEVEL_PRED-1
data_test$TRAJ_COUNT_LEVEL <- TRAJ_COUNT_LEVEL_PRED





#######################
# PREDICT TRAJ_LENGTH #
#######################

formula <- ~ LOG_STRAIGHT_DIST + LOG_MANHATTAN_DIST + LOG_DRIVER_LONGCUT + X_START +
              Y_START + X_END + Y_END + HOUR_N + WEEKEND + BEARING + TRAJ_COUNT_LEVEL

x <- sparse.model.matrix(formula, data_train)[,-1]
y <- data_train$LOG_TRAJ_LENGTH

# Final Optimal Parameters for Traj Length XGBoost Prediction Model
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3)
xgb.grid <- expand.grid(nrounds = 5000,
                        max_depth = c(20, 24),
                        eta = 0.1,
                        gamma = 0.0,
                        colsample_bytree = 0.5,
                        subsample = 1.0,
                        min_child_weight = c(1, 25))
xgb_tune2 <- train(x, y, method="xgbTree", metric = "RMSE", trControl=cv.ctrl, tuneGrid=xgb.grid)
xgb_tune2$bestTune

# Create Traj Length XGBoost Prediction Model
dtrain <- xgb.DMatrix(data = x, label = y)
params <- list( booster = "gbtree", 
                objective = "reg:linear", 
                eval_metric = "rmse", 
                eta=0.01,
                colsample_bytree=0.5, 
                min_child_weight=1, 
                max_depth=20)

xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                nrounds = 10000, 
                nfold = 5, 
                showsd = T, 
                stratified = T, 
                print.every.n = 10, 
                early.stop.round = 20, 
                maximize = F)

xgbfit <- xgboost(params = params, data = dtrain, nrounds =xgbcv$best_iteration)

# Predict Traj Length in Test Data
x <- sparse.model.matrix(formula, data_test)[,-1]
TRAJ_LENGTH_PRED <- exp(predict(xgbfit, newdata = x))
data_test$LOG_TRAJ_LENGTH <- log(TRAJ_LENGTH_PRED)





####################
# PREDICT DURATION #
####################
formula <- ~ LOG_TRAJ_LENGTH + LOG_DRIVER_SPEED_INV + X_START + Y_START + X_END + Y_END +
              HOUR_N + DAY_OF_WEEK_N + RUSH_HOUR + WEEKEND + BEARING + TRAJ_COUNT_LEVEL

x <- sparse.model.matrix(formula, data_train)[,-1]
y <- data_train$LOG_DURATION

# Final Optimal Parameters for Duration XGBoost Prediction Model
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3)
xgb.grid <- expand.grid(nrounds = 3000,
                        max_depth = c(10, 16),
                        eta = 0.1,
                        gamma = 0.0,
                        colsample_bytree = 0.5,
                        subsample = 1.0,
                        min_child_weight = c(100, 120)
)
xgb_tune3 <-train(x, y, method="xgbTree", metric = "RMSE", trControl=cv.ctrl, tuneGrid=xgb.grid)
xgb_tune3$bestTune

# Create Duration XGBoost Prediction Model
dtrain <- xgb.DMatrix(data = x, label = y)
params <- list( booster = "gbtree", 
                objective = "reg:linear", 
                eval_metric = "rmse", 
                eta=0.01, 
                colsample_bytree=0.5, 
                min_child_weight=120, 
                max_depth=10)

xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                nrounds = 10000, 
                nfold = 5, 
                showsd = T, 
                stratified = T, 
                print.every.n = 10, 
                early.stop.round = 20, 
                maximize = F)

xgbfit <- xgboost(params = params, data = dtrain, nrounds =xgbcv$best_iteration)

# Predict Duration in Test Data
x <- sparse.model.matrix(formula, data_test)[,-1]
DURATION_PRED <- exp(predict(xgbfit, newdata = x))





######################
# PREPARE SUBMISSION #
######################

submission = data.frame(ID = c(465173:930344), PRICE = TRAJ_LENGTH_PRED + DURATION_PRED)
colnames(submission) <- c("ID", "PRICE")
write.csv(x=submission, 'submission.csv', row.names = FALSE)
