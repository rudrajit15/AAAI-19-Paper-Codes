function [] = predict_alpha()
gamma = 15;
n0 = 50;
n1 = 110;
n2 = 20;
m = 5500;
n = n0*n1+n1*n2;
sigma = 0.2355;
Lc = 0.4626;
beta = 0.001;

%mu_s = sqrt(2*m)/(n0*n1*sigma*sqrt(exp(-Lc)*(1-exp(-Lc))));
mu_s = sqrt(2*m*(Lc+1)*(2*Lc+1))/(sigma*sqrt(n0*n1*Lc));
lambda_max_old = 0.4*(sqrt(n0*n1)+sqrt(n1*n2))*(1-exp(-Lc))/sqrt(2*m*n2)
lambda_max = 0.45*(sqrt(n0*n1)+sqrt(n1*n2))*Lc/sqrt(m*n2*(2*Lc+1)*(Lc+1))

mu_1 = mu_s*sqrt(beta);
mu_2 = mu_s*sqrt(lambda_max);

c1 = sqrt(pi/(4*gamma))*((exp(gamma+mu_1))/(exp(gamma)-1))*mu_2*exp(mu_2^2/(4*gamma));
c2 = erf(sqrt(gamma)+mu_2/(2*sqrt(gamma))) - erf(sqrt(gamma)*sqrt(beta/lambda_max)+mu_2/(2*sqrt(gamma)));
c3 = (1-exp(-(mu_2-mu_1)))/(exp(gamma)-1);

c4 = (exp(gamma*(1-beta/lambda_max))-1)/(exp(gamma)-1);

alpha = (m/(2*n))*(c1*c2-c3) + ((n-m)/(2*n))*c4

x = linspace(sqrt(beta),sqrt(lambda_max),1000);
c = 1/lambda_max;
z = (exp(gamma*c*(lambda_max-x.^2))-1)/(exp(gamma*c*lambda_max)-1);
y = mu_s*exp(-(mu_s*(x-sqrt(beta)))).*z;
ar = trapz(x,y);

alpha2 = (m/(2*n))*ar + ((n-m)/(2*n))*(exp(gamma*c*(lambda_max-beta))-1)/(exp(gamma)-1)

