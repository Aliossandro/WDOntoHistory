library(clValid)
library(diceR)

setwd('~/Documents/PhD/OntoHistory/')

userdata <- read.csv('userData.csv')

userdata[is.na(userdata)] <-0
userdata$oldedit <- userdata$oldedit/360

clusters <- kmeans(userdata[,2:9], 4, iter.max = 20, nstart = 16,
       algorithm = c("Hartigan-Wong"), trace=FALSE)

userdata$cluster <- clusters$cluster


compactness(userdata[,2:9], userdata[,10])


intern <- clValid(userdata[,2:9], 2:8, clMethods="kmeans",
                  validation="internal")


wilcox.test(userdata$oldedit[which(userdata$cluster == 4)], userdata$oldedit[which(userdata$cluster == 2)])

d <-  density(log(userdata$oldedit))
plot(d)