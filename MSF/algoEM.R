algoEM <- function(x, e) {
	m <- mean(x);
	var <- var(x);

	pn <- .5;
	m1n <- 2*m;
	var1n <- .5*var;
	m2n <- -3*m;
	var2n <- 2*var;

	thetan <- c(pn, m1n, var1n, m2n, var2n);
	ecart <- 10;
	count<- 0;
	while(ecart>e){

		phi1 <- dnorm(x, thetan[2], sqrt(thetan[3]));
		phi2 <- dnorm(x, thetan[4], sqrt(thetan[5]));

		p1Rt <- ( thetan[1] * phi1 ) / ( thetan[1] * phi1 + (1-thetan[1]) * phi2 ); 
		p0Rt <- ( (1-thetan[1]) * phi2 ) / ( thetan[1] * phi1 + (1-thetan[1]) * phi2 );

		pn1 <- (1/length(p1Rt)) * sum(p1Rt);
		m1n1 <- (p1Rt%*%x)/sum(p1Rt);
		var1n1 <- (p1Rt%*%(x*x))/sum(p1Rt) - (thetan[2]*thetan[2]);
		m2n1 <- (p0Rt%*%x)/sum(p0Rt);
		var2n1 <- (p0Rt%*%(x*x))/sum(p0Rt) - (thetan[4]*thetan[4]);

		thetan1 <- c(pn1, m1n1, var1n1, m2n1, var2n1);

		ecart <- abs( (sqrt(thetan1%*%thetan1) - sqrt(thetan%*%thetan))/sqrt(thetan%*%thetan) )
		thetan <- thetan1;
		count<-count+1;
	}
	return (c(thetan,count));
}
a<-algoEM(RentJ$Rt, 10^(-6));	

#Res<-HMMFIT(RentJ$Rt)
#matrice de transition, Res$HMM $distribution $mean $var


