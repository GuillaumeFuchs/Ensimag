data<-read.csv("data.csv", sep=";", dec=".")
colnames(data)<-c("code", "year", "month", "beta", "fsrv", "capit", "btm", "turnover", "renta")
data2<-read.csv("data_facteurs.csv", sep=";", dec=".")
colnames(data2)<-c("year", "month", "rmrf", "smb", "hml", "umd", "rf")


data<-data[data[,"year"]==1999 & data[,"month"]<=9,] #on prend les titres sur la période de janvier à mars 1999 pour la période de avril à septembre 1999 car période ou tous les titres sont présents
data2<-data2[data2[,"year"]==1999 & data2[,"month"]<=9,]

for (t in (1:9))
{
	data[data[,"month"]==t, "renta"] = data[data[,"month"]==t, "renta"] + data2[data2[,"month"]==t, "rf"]
}

transact = 0.001 #cout de transaction

#########
###CALCUL DE LA RENTABILITE CUMULEE DE CHAQUE ACTIF SUR LES 3 MOIS
#########
rc<-matrix(nrow=300, ncol=2) #vecteur contenant les rentabilites cumulees des 300 actifs
colnames(rc) <- c("code", "renta")

for (i in (1:300))
{
	r = 1
	rc[i, 1] <- i

	for (j in (1:3))
	{
		r = r*(1+data[data[,"code"]==i & data[,"month"]==j, "renta"])
	}
	rc[i, 2] <- r-1
}
#tri
rc<-rc[order(rc[,2]),]

#renta cumulee du taux sans risque
rcrf = 1
for (t in (4:9))
{
	rcrf = rcrf * (1 + data2[data2[,"month"]==t,"rf"])
}
rcrf = rcrf - 1

#########
###CALCUL DU PORTEFEUILLE DE MEILLEUR ET DE PLUS MAUVAISE RENTABILITE CUMULEE SUR 3 MOIS
#########

#selection des actifs
compoPfM <- rc[(1:30) ,1] #contient les 30 actifs avec les rentabilites cumulee les plus faible
compoPfP <- rc[(271:300),1]  #contient les 30 actifs les plus

pfM <- matrix(nrow=8, ncol=124) #contient l'ensemble des informations sur le portefeuille MOINS
pfP <- matrix(nrow=8, ncol=124) #idem pour portefeuille PLUS
rownames(pfM) <- c("t=0", "t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "cumulee")
rownames(pfP) <- c("t=0", "t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "cumulee")
nameM <- c()
nameP <- c()
for (i in compoPfM)
{
	nameM <- c(nameM, i, "value", "quantity", "proportion")
}
for (i in compoPfP)
{
	nameP <- c(nameP, i, "value", "quantity","proportion")
}
nameM <- c(nameM, "R", "value before", "value after", "transactions cost")
nameP <- c(nameP, "R", "value before", "value after", "transactions cost")
colnames(pfM) <- nameM
colnames(pfP) <- nameP

#########
###CALCUL DES RENTABILITES ET DE LA COMPOSITION DU PORTEFEUILLE SUR 6 MOIS
#########
for (t in (1:7))
{
	pfM[t, "R"] <- 0
	pfP[t, "R"] <- 0
	
	pfM[t, "value before"] <- 0
	pfP[t, "value before"] <- 0

	pfM[t, "value after"] <- 0
	pfP[t, "value after"] <- 0

	sum_asset_m = 0
	sum_asset_p = 0

	#calcul des rentabilites et de la valeur de chaque actif
	#calcul de la valeur du portefeuille sans rebalancement
	indice = 1
	for (i in compoPfM)
	{
		if (t == 1)
		{
			pfM[t, indice] <- 0
			pfM[t, indice+1] <- 100
			pfM[t, indice+2] <- 1
			pfM[t, "value before"] <- pfM[t, "value before"] + pfM[t, indice+1]
			pfM[t, "value after"] <- pfM[t, "value before"]
		}else
		{
			pfM[t, indice] <- data[data[,"code"]==i & data[,"month"]==t+2, "renta"] #rentabilite
			pfM[t, indice+1] <- (1+pfM[t, indice]) * pfM[t-1, indice+1] #valeur
			sum_asset_m = sum_asset_m + pfM[t, indice+1] #somme des valeurs des actifs avant rebalancement
			pfM[t, "value before"] <- pfM[t, "value before"] + pfM[t-1, indice+2]*pfM[t, indice+1] #valeur du portefeuille avec les quantites en t-1 (avant rebalancement)
		}
		indice = indice + 4
	}
	
	indice = 1
	for (i in compoPfP)
	{
		if (t == 1)
		{
			pfP[t, indice] <- 0
			pfP[t, indice+1] <- 100
			pfP[t, indice+2] <- 1
			pfP[t, "value before"] <- pfP[t, "value before"] + pfP[t, indice+1]
			pfP[t, "value after"] <- pfP[t, "value before"]
		}else
		{
			pfP[t, indice] <- data[data[,"code"]==i & data[,"month"]==t+2, "renta"]
			pfP[t, indice+1] <- (1+pfP[t, indice]) * pfP[t-1, indice+1]
			sum_asset_p = sum_asset_p + pfP[t, indice+1]
			pfP[t, "value before"] = pfP[t, "value before"] + pfP[t-1, indice+2]*pfP[t, indice+1]
		}
		indice = indice + 4
	}
	
	#calcul de la quantite de chaque actif
	#calcul de la valeur du portefeuille apres rebalancement
	indice = 1
	if (t != 1)
	{
		for (i in compoPfM)
		{
			pfM[t, indice+2] <- (1 + transact)*pfM[t, "value before"] / (pfM[t, indice+1]*(length(compoPfM) + transact*(sum_asset_m - pfM[t, indice+1] + 1)))
			pfM[t, "value after"] <- pfM[t, "value after"] + pfM[t, indice+2] * pfM[t, indice+1]
			pfM[t, "R"] <- pfM[t, "value after"]/pfM[t-1, "value after"] - 1 

			pfP[t, indice+2] <- (1 + transact)*pfP[t, "value before"] / (pfP[t, indice+1]*(length(compoPfP) + transact*(sum_asset_p - pfP[t, indice+1] + 1)))
			pfP[t, "value after"] <- pfP[t, "value after"] + pfP[t, indice+2] * pfP[t, indice+1]
			pfP[t, "R"] <- pfP[t, "value after"]/pfP[t-1, "value after"] - 1 
			indice = indice + 4
		}
	}

	#calcul des proportions de chaque actif dans le portefeuille
	indice = 1
	for (i in compoPfM)
	{
		pfM[t, indice+3] <- pfM[t, indice+2]*pfM[t, indice+1]/pfM[t, "value after"]*100
		pfP[t, indice+3] <- pfP[t, indice+2]*pfP[t, indice+1]/pfP[t, "value after"]*100
		indice = indice + 4
	}

	#Calcul des couts de transaction
	pfM[t, "transactions cost"] <- pfM[t,"value before"] - pfM[t, "value after"]
	pfP[t, "transactions cost"] <- pfP[t,"value before"] - pfP[t, "value after"]
}


#Calcul de la rentabilite cumulée de chaque actif et du pf
indice = 1
for (i in compoPfM)
{
	rm = 1
	rp = 1
	for (t in (2:7))
	{
		rm = rm * (1+pfM[t, indice])
		rp = rp * (1+pfP[t, indice])
	}
	pfM[8, indice] <- rm - 1
	pfP[8, indice] <- rp - 1
	indice = indice + 4
}

rm = 1
rp = 1
for (t in (2:7))
{
	rm = rm * (1+pfM[t, "R"])
	rp = rp * (1+pfP[t, "R"])
}
pfM[8, "R"] <- rm - 1
pfP[8, "R"] <- rp - 1


#########
####CALCUL PORTEFEUILLE: ACHAT DU PORTEFEUILLE DE MEILLEUR RENTABILITE ET VENTE DU PORTEFEUILLE DE PLUS MAUVAISE RENTABILITE
#########

#########
####MESURE DE PERFORMANCE
#########
#Calcul de la variance
varpp = 0
varpm = 0
for (t in (1:6))
{
	varpp = (varpp - mean(pfP[(2:7), "R"]))*(varpp - mean(pfP[(2:7), "R"]))
	varpm = (varpm - mean(pfM[(2:7), "R"]))*(varpm - mean(pfM[(2:7), "R"]))
}
varpp = (1/6)*varpp
varpm = (1/6)*varpm

#ratio de sharp
#si ratio negatif, le pf a moins perf que le taux sans risque => tres mauvais
#si entre 0 1, le risque pris est trop eleve pour le rendement obtenu
#si sup a 1, la sur performance du pf par rapport au taux sans risque ne se fait pas au prix d'un risque trop eleve
sharpp = (pfP["cumulee", "R"] - rcrf)/ varpp
sharpm = (pfM["cumulee", "R"] - rcrf)/ varpm