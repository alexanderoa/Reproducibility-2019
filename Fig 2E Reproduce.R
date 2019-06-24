library(tidyverse)
library(reshape)
library(poibin)

nhanes <- read_csv('nhanes_labs2.csv')
nhanes_cont <- select_if(nhanes, is.numeric) #Subset for only continuos data
nhanes_cont <- nhanes_cont[,colSums(is.na(nhanes_cont))<(nrow(nhanes_cont))/2] #Keep only data where more than half the rows are not NA

cols <- t( combn(colnames(nhanes_cont), 2) ) #All pairwise relationships from dataset
nhanes_pval <- apply( cols , 1 , function(x){ #Calculated p-values
  first <- nhanes_cont[x[1]]
  second <- nhanes_cont[x[2]]
  v1 <- unlist(first, use.names = FALSE)
  v2 <- unlist(second, use.names = FALSE)
  cor.test( v1 , v2, method = 'pearson')$p.value 
} )

nhanes_bonfer <- p.adjust(nhanes_pval, method = 'bonferroni') #Adjust p-values

null_list <- cbind(cols, nhanes_bonfer) 

v <- sapply(null_list[, 3], function(x) x < 0.05) #Determine if relationships are null or non-null

null_list <- cbind(null_list, v)

n2 <- length(v) - sum(v) #Number of null relationships
n1 <- sum(v) #Number of non-null relationships

u = 0.1 #bias term

dPPV_mat <- matrix(nrow = 100, ncol = 100) #matrix to hold simulations of dPPV

for (k in 1:100){
  teams <- matrix(runif(148500), ncol = 1485) #matrix of teams and relationships (in this case 100 teams)
  #random numbers are used to determine if a team studies a relationship
  S <- apply(teams, c(1,2), function(x) as.numeric(x > 0.95)) #if an entry S(i,j) > 0.95, team i studies relationship j
  
  B <- apply(S, c(1,2), function(x){ #generate betas for all studied team-relationship pairs
    if(x == 1){
      x <- runif(1, min = 0.05, max = 0.25)
    }
    else{
      x = 0
    }
  })
  
  non_null <- 1:276 #vector to hold indices of non_null relationships
  j = 0
  for (i in 1:1485){
    if (v[i] == TRUE) {
      non_null[j] = i
      j = j+1
    }
  }
  
  null <- 1:1209 #vector to hold indices of null relationships
  j = 0
  for (i in 1:1485){
    if (v[i] == FALSE) {
      non_null[j] = i
      j = j+1
    }
  }
  
  pp <- 1:276 #vector of p_i
  
  for (i in 1:276){
    hold <- 1 - u
    for (j in 1:100){
      hold <- hold*B[j,non_null[i]]^(S[j,non_null[i]])
    }
    pp[i] <- 1 - hold
  }
  
  qq <- 1:1209 #vector of q_i
  
  for (i in 1:1209){
    hold <- 1 - u
    for (j in 1:100){
      hold <- hold*(1-0.05)^(S[j,null[i]])
    }
    qq[i] <- 1 - hold
  }
  
  P <- rpoibin(100, pp) #randomly generate values from P
  Q <- rpoibin(100, qq)
  
  dPPV <- P/(P+Q) 
  
  dPPV_mat[k,] = dPPV
}

dPPV_vec <- as.vector(dPPV_mat)
dPPV_dat <- as.data.frame(dPPV_vec)

ggplot(data = dPPV_dat, aes(x=dPPV_vec)) + geom_density() + xlim(0,1)





