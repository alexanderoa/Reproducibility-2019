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
  
  non_null <- 1:n1 #vector to hold indices of non_null relationships
  j = 0
  for (i in 1:1485){
    if (v[i] == TRUE) {
      non_null[j] = i
      j = j+1
    }
  }
  
  null <- 1:n2 #vector to hold indices of null relationships
  j = 0
  for (i in 1:1485){
    if (v[i] == FALSE) {
      non_null[j] = i
      j = j+1
    }
  }
  
  pp <- 1:n1 #vector of p_i
  
  for (i in 1:n1){
    hold <- 1 - u
    for (j in 1:100){
      hold <- hold*B[j,non_null[i]]^(S[j,non_null[i]])
    }
    pp[i] <- 1 - hold
  }
  
  qq <- 1:n2 #vector of q_i
  
  for (i in 1:n2){
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

##################################################################################

trial_list <- c()
for (i in 1:100){ #variance increases as number of trials increases
  num_sample <- as.integer(runif(1, 5, 50))
  sample_index <- sample(1:nrow(nhanes_cont), num_sample)
  nhanes_trial <- nhanes_cont[sample_index,]
  nhanes_trial <- nhanes_trial[,colSums(is.na(nhanes_trial))<1*(nrow(nhanes_trial))/4]
  
  cols <- t( combn(colnames(nhanes_trial), 2) )
  nhanes_trial_pval <- apply( cols , 1 , function(x){
    first <- nhanes_trial[x[1]]
    second <- nhanes_trial[x[2]]
    v1 <- unlist(first, use.names = FALSE)
    v2 <- unlist(second, use.names = FALSE)
    cor.test( v1 , v2, method = 'pearson')$p.value 
  } )
  nhanes_trial_bon<- p.adjust(nhanes_trial_pval, method = 'bonferroni')
  
  null_list2 <- cbind(cols, nhanes_trial_bon)
  
  v2 <- sapply(null_list2[, 3], function(x) x < 0.05)
  
  trial_list <- c(trial_list, which(v %in% TRUE))
}


u = 0.1

dPPV_mat2 <- matrix(nrow = 100, ncol = 100)

for (k in 1:100){
  teams <- matrix(runif(148500), ncol = 1485)
  for (i in 1:length(trial_list)){
    for (j in 1:100){
      if (teams[j, trial_list[i]] > 0.8){
        teams[j, trial_list[i]] == 1
      }
    }
  }
  S <- apply(teams, c(1,2), function(x) as.numeric(x > 0.99))
  
  B <- apply(S, c(1,2), function(x){
    if(x == 1){
      x <- runif(1, min = 0.05, max = 0.25)
    }
    else{
      x = 0
    }
  })
  
  non_null <- 1:n1
  j = 0
  for (i in 1:1485){
    if (v[i] == TRUE) {
      non_null[j] = i
      j = j+1
    }
  }
  
  null <- 1:n2
  j = 0
  for (i in 1:1485){
    if (v[i] == FALSE) {
      non_null[j] = i
      j = j+1
    }
  }
  
  pp <- 1:n1
  
  for (i in 1:n1){
    hold <- 1 - u
    for (j in 1:100){
      hold <- hold*B[j,non_null[i]]^(S[j,non_null[i]])
    }
    pp[i] <- 1 - hold
  }
  
  qq <- 1:n2
  
  for (i in 1:n2){
    hold <- 1 - u
    for (j in 1:100){
      hold <- hold*(1-0.05)^(S[j,null[i]])
    }
    qq[i] <- 1 - hold
  }
  
  P <- rpoibin(100, pp)
  Q <- rpoibin(100, qq)
  
  dPPV <- P/(P+Q)
  
  dPPV_mat2[k,] = dPPV
}

dPPV_vec2 <- as.vector(dPPV_mat2)
dPPV_dat <- data.frame(v1=dPPV_vec, v2=dPPV_vec2)

data <- melt(dPPV_dat) 
ggplot(data = data, aes(x=value, fill=variable)) + geom_density(alpha=0.5) + scale_fill_discrete(labels = c("w/o trials", "w/ trials"))

