library(clValid)
library(diceR)
library(cluster)
library(factoextra)
# library(NbClust)
setwd('~/Documents/PhD/userstats/')


userdata <- read.csv('frameClean.csv', stringsAsFactors = F)
userdata <- userdata[,-1]
userdata$admin[which(userdata$admin == 'True')] = 0
userdata$admin[which(userdata$admin == 'False')] = 1
userdata$admin <- as.numeric(userdata$admin)

userdata$lowAdmin[which(userdata$lowAdmin == 'True')] = 0
userdata$lowAdmin[which(userdata$lowAdmin == 'False')] = 1
userdata$lowAdmin <- as.numeric(userdata$lowAdmin)

# userdata <- userdata[which(userdata$labels < 1),]
# userdata$admin <-  as.logical(userdata$admin)
# userdata <- userdata[,-(11:14)]
userdata <- subset(userdata, select = -noEdits)
# userdata <- subset(userdata, select = -normAll)
# userdata <- subset(userdata, select = -username)
userdata <- subset(userdata, select = -timeframe)
userdata <- subset(userdata, select = -serial)
# userdata <- subset(userdata, select = -labels)

userdata[is.na(userdata)] <-0



# K-Means Clustering with 5 clusters
fit <- kmeans(userdata, 2)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster)
clusplot(userdata, fit$cluster, color=TRUE, shade=TRUE,
         labels=2, lines=0)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(userdata, fit$cluster) 
# userdata <- subset(userdata, select = -noEdits)
userdata_sample <- userdata[sample(nrow(userdata), 25000), ]
gskmn2 <- clusGap(userdata, FUN = kmeans, nstart = 10, K.max = 8, B = 60, spaceH0="original")

fviz_gap_stat(gskmn2, 
              maxSE = list(method = "Tibs2001SEmax"))

cosi <- kmeans(userdata, 2, iter.max = 10, nstart = 6, trace=FALSE)
userdata$clusters <-  cosi$cluster
userdata

compactness(userdata[,2:9], userdata[,10])

###validation
result_intern <- vector("list", 5)
result_stab <- vector("list", 5)

intern <- function(x) {
  
  userdata_sample <- x[sample(nrow(x), 10000), ]
  y <- clValid(userdata_sample[,1:9], 2:9, clMethods="kmeans",
               validation="internal", maxitems = nrow(userdata_sample))
  
  return(y)
  # fviz_nbclust(nb)
}
stabo <- function(x) {
  userdata_sample <- x[sample(nrow(x), 10000), ]
  y <- clValid(userdata_sample[,1:9], 2:9, clMethods="kmeans",
               validation="stability", maxitems = nrow(userdata_sample))
  
  return(y)
  
}

coso <-  replicate(5, intern(userdata))
coso2 <-  replicate(5, stabo(userdata))


wilcox.test(userdata$oldedit[which(userdata$cluster == 4)], userdata$oldedit[which(userdata$cluster == 2)])

d <-  density(log(userdata$oldedit))
plot(d)