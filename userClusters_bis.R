library(clValid)
library(diceR)

setwd('~/Documents/PhD/WDOntoHistory/')

userdata <- read.csv('userData.csv')

userdata[is.na(userdata)] <-0
userdata$oldedit <- userdata$oldedit/360

userdata <- userdata[which(userdata$bot_label == 0),]
userdata <- userdata[which(userdata$anon_label == 0),]
userdata <-  subset(userdata, select = - bot_label)
userdata <-  subset(userdata, select = - anon_label)

clusters <- kmeans(userdata[,2:7], 3, iter.max = 20, nstart = 16,
       algorithm = c("Hartigan-Wong"), trace=FALSE)

userdata$cluster <- clusters$cluster


compactness(userdata[,2:9], userdata[,10])


intern <- clValid(userdata[,2:7], 2:8, clMethods="kmeans",
                  validation="internal", maxitems=nrow(userdata))


wilcox.test(userdata$oldedit[which(userdata$cluster == 4)], userdata$oldedit[which(userdata$cluster == 2)])

d <-  density(log(userdata$oldedit))
plot(d)