library(tidyverse)
library(reshape)
library(poibin)

nhanes <- read_csv('nhanes_labs2.csv')
nhanes_cont <- select_if(nhanes, is.numeric)
nhanes_cont <- nhanes_cont[,colSums(is.na(nhanes_cont))<(nrow(nhanes_cont))/2]

nhanes_cor <- cor(nhanes_cont, method = 'pearson', use = 'pairwise')

cols <- t( combn(colnames(nhanes_cont), 2) )
nhanes_pval <- apply( cols , 1 , function(x){
  first <- nhanes_cont[x[1]]
  second <- nhanes_cont[x[2]]
  v1 <- unlist(first, use.names = FALSE)
  v2 <- unlist(second, use.names = FALSE)
  cor.test( v1 , v2, method = 'pearson')$p.value 
} )

nhanes_bonfer <- p.adjust(nhanes_pval, method = 'bonferroni')

null_list <- cbind(cols, nhanes_bonfer)

v <- sapply(null_list[, 3], function(x) x < 0.05)

null_list <- cbind(null_list, v)

n2 <- length(v) - sum(v)
n1 <- sum(v)

u = 0.05

dPPV_mat <- matrix(nrow = 100, ncol = 100)

for (k in 1:100){
  teams <- matrix(runif(148500), ncol = 1485)
  S <- apply(teams, c(1,2), function(x) as.numeric(x > 0.95))
  
  B <- apply(S, c(1,2), function(x){
    if(x == 1){
      x <- runif(1, min = 0.05, max = 0.25)
    }
    else{
      x = 0
    }
  })
  
  non_null <- 1:276
  j = 0
  for (i in 1:1485){
    if (v[i] == TRUE) {
      non_null[j] = i
      j = j+1
    }
  }
  
  null <- 1:1209
  j = 0
  for (i in 1:1485){
    if (v[i] == FALSE) {
      non_null[j] = i
      j = j+1
    }
  }
  
  pp <- 1:276
  
  for (i in 1:276){
    hold <- 1 - u
    for (j in 1:100){
      hold <- hold*B[j,non_null[i]]^(S[j,non_null[i]])
    }
    pp[i] <- 1 - hold
  }
  
  qq <- 1:1209
  
  for (i in 1:1209){
    hold <- 1 - u
    for (j in 1:100){
      hold <- hold*(1-0.05)^(S[j,null[i]])
    }
    qq[i] <- 1 - hold
  }
  
  P <- rpoibin(100, pp)
  Q <- rpoibin(100, qq)
  
  dPPV <- P/(P+Q)
  
  dPPV_mat[k,] = dPPV
}

dPPV_vec <- as.vector(dPPV_mat)
dPPV_dat <- as.data.frame(dPPV_vec)

ggplot(data = dPPV_dat, aes(x=dPPV_vec)) + geom_density() + xlim(0,1)





