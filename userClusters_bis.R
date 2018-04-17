library(clValid)
library(diceR)
library(cluster)
library(factoextra)
library(clv)
# library(NbClust)
setwd('~/Documents/PhD/userstats/')


userdata <- read.csv('frameClean.csv', stringsAsFactors = F)
# userdata <- userdata[,-1]
# userdata <- userdata[which(userdata$labels < 2),]
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
userdata <- subset(userdata, select = -username)
userdata <- subset(userdata, select = -timeframe)
userdata <- subset(userdata, select = -serial)
# userdata <- subset(userdata, select = -labels)

userdata[is.na(userdata)] <-0

wss_value <- list()
conn_value <- list()
for (j in 1:8) {
userdata_sample <- userdata[sample(nrow(userdata), 626883), ]
fit <-  kmeans(userdata, 1, iter.max = 20, nstart = 16,
       algorithm = c("Hartigan-Wong"), trace=FALSE)

###wss
wss <- (nrow(userdata)-1)*sum(apply(userdata,2,var))
conn <- clv::connectivity(userdata, fit$cluster, 10)
# comp <-  compactness(userdata_sample, kmeans(userdata_sample, 1, iter.max = 20, nstart = 16,
#                                              algorithm = c("Hartigan-Wong"), trace=FALSE)$cluster)
for (i in 2:8) wss[i] <- sum(kmeans(userdata,centers=i)$withinss)
for (i in 2:8) conn[i] <-clv::connectivity(userdata, kmeans(userdata, i, iter.max = 20, nstart = 16,algorithm = c("Hartigan-Wong"), trace=FALSE)$cluster, 10)
wss_value[[j]] <- wss
conn_value[[j]] <- conn
}

connVector <- lapply(conn_value,function(x) as.vector(x))
connDf <- as.data.frame(do.call(rbind, connVector))
connDf <-  apply(connDf, 2, function(x) mean(x))

wssVector <- lapply(wss_value,function(x) as.vector(x))
wssDf <- as.data.frame(do.call(rbind, wssVector))
wssDf <-  apply(wssDf, 2, function(x) mean(x))

plot(connDf, wssDf, type="b")
plot(conn, wss, type="b")


# for (i in 2:9) comp[i] <- compactness(userdata_sample, kmeans(userdata_sample, i, iter.max = 20, nstart = 16,
#                                                               algorithm = c("Hartigan-Wong"), trace=FALSE)$cluster)
resulti <- as.data.frame(wss,conn)


plot(1:9, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")



# K-Means Clustering with 5 clusters
fit <- kmeans(userdata, 4)

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

cosi <- kmeans(userdata, 4, iter.max = 10, nstart = 6, trace=FALSE)
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