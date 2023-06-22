library("nlme")
library("nnet")
library("survival")
library("BART")

set.seed(99)
for (i in 0:4) {
    # Load trainig data
    x.train <- read.csv(file = sprintf("boston_x-train_%i.csv", i), sep=",")
    y_ <- read.csv(file = sprintf("boston_y-train_%i.csv", i), sep=",")
    y.train <- y_$medv# using the column as numeric class
    # Load test data
    x.test <- read.csv(file =sprintf("boston_x-test_%i.csv", i), sep=",")
    y2_ <- read.csv(file =sprintf("boston_y-test_%i.csv", i), sep=",")
    y.test <- y2_$medv# using the column as numeric class

    # Run 4 chain for each dataset
    for (j in 1:4) {
        nd <- 1000
        burn <- 1000
        post <- wbart(x.train, y.train, nskip = burn, ndpost = nd)

        # Save trace data
        end <- burn+nd
        column1 <- post$sigma[(burn+1):end]
        column2 <- post["yhat.train"]
        df <- data.frame(column1, column2)
        write.table(df, file = sprintf("trace_R-BART_0%i_%i.csv", i, j), sep = ",")

        # calculate predictions for each test
        yhat <- predict(post, x.test)
        if (j==1){
            y_new <- yhat
        } else {
            y_new <- rbind(y_new, yhat)
        }
    }
    # Save all predictions in one ("4 chains")
    write.table(y_new, file=sprintf("y_pred_R-BART_0%i.csv", i), sep=",")
}
