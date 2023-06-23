library("MASS")

library("nlme")
library("nnet")
library("survival")

library("BART")

# Two variables
x <- Boston[, c(6, 13)]

y <- Boston$medv

set.seed(99)
for (j in 1:4) {
    nd <- 1000
    burn <- 1000
    post <- wbart(x, y, nskip = burn, ndpost = nd)
    # Save data
    end <- burn+nd
    column1 <- post$sigma[(burn+1):end]
    column2 <- post["yhat.train"]
    df <- data.frame(column1, column2)
    write.table(df, file = sprintf("boston_R-BART_0%i.csv", j), sep = ",")
}


# All variables
x_full <- Boston[, 1:13]

set.seed(99)
for (j in 1:4) {
    nd <- 1000
    burn <- 1000
    post <- wbart(x_full, y, nskip = burn, ndpost = nd)
    # Save data
    end <- burn+nd
    column1 <- post$sigma[(burn+1):end]
    column2 <- post["yhat.train"]
    df <- data.frame(column1, column2)
    write.table(df, file = sprintf("boston_R-BART_0%i_full.csv", j), sep = ",")
}
