function [eigvals,eigvals2,eigvals_sum,diff,H] = sim_Hessian3(loss)
%function [diff] = sim_Hessian3(loss)
% W1 is n1*n0, W2 is 1*n1
% m = no. of training examples
% x,W1,W2 are assumed to be normally distributed
%n0 = 51;
%n1 = 101;

n0 = 50;
n1 = 110;
n2 = 20;
m = 3600;

%n0 = 15;
%n1 = 30;
%n2 = 5;
%m = 650;

%n0 = 21;
%n1 = 81;
%n2 = 10;
%m = 3000;
% std of weights
%sigma = 0.3454;
%sigma = 0.4074;
%sigma = 0.5123;
%sigma = 0.4218;
sigma =  0.3883;
x = randn(n0,m);
%W1 = normrnd(0,0.4118,[n1,n0]);
%W2 = normrnd(0,0.5915,[n2,n1]);
%%W1 = normrnd(0,0.3246,[n1,n0]);
%%W2 = normrnd(0,0.4645,[n2,n1]);
W1 = normrnd(0,sigma,[n1,n0]);
W2 = normrnd(0,sigma,[n2,n1]);
z = W1*x;
% ReLU 
z(z<0) = 0;
perm = randperm(m);
t = m/n2;
y = zeros(n2,m);
for i = 1:n2
    y(i,perm((i-1)*t+1:i*t)) = 1;
end
y_pred = y;
acc = 1;
for i = 1:m
    yi = y(:,i);
    max_pos = find(yi == 1);
    ch = binornd(1,acc);
    %max_prob = (0.5 + rand/2);
    max_prob = exp(-loss);
    y_pred(:,i) = (1-max_prob)/(n2-1);
    if(ch == 1)
        y_pred(max_pos,i) = max_prob;
    else
        idx = 1:n2;
        idx(max_pos) = [];
        y_pred(idx(randi(n2-1)),i) = max_prob;
    end
end
err = normrnd(0,1,[n2,m]);
params = n1*n0+n2*n1
H = zeros(params,params);
H2 = zeros(params,params);
% mapping : W1(k,j) = n0*(k-1)+j 
for a = 1:params
    for b = a:params
        % both belonging to W1
        if((a<=n1*n0) && (b<=n1*n0))
            rem1 = mod(a,n0);
            if(rem1 == 0)
                k1 = a/n0;
                j1 = n0;
            else
                k1 = (a-rem1)/n0 + 1;
                j1 = rem1;
            end
            rem2 = mod(b,n0);
            if(rem2 == 0)
                k2 = b/n0;
                j2 = n0;
            else
                k2 = (b-rem2)/n0 + 1;
                j2 = rem2;
            end
            
            tmp = W2(:,k1).*W2(:,k2).*y.*y_pred.*(1-y_pred);
            tmp = sum(tmp,1)';
            tmp = tmp.*(x(j1,:)').*(x(j2,:)');
            tmp = tmp.*(double(z(k1,:)>0)').*(double(z(k2,:)>0)');
            H(a,b) = sum(tmp);
        end
        
        % both belonging to W2
        if((a>n1*n0) && (b>n1*n0))
            a2 = a - n1*n0;
            b2 = b - n1*n0;
            rem1 = mod(a2,n1);
            if(rem1 == 0)
                l1 = a2/n1;
                k1 = n1;
            else
                l1 = (a2-rem1)/n1 + 1;
                k1 = rem1;
            end
            rem2 = mod(b2,n1);
            if(rem2 == 0)
                l2 = b2/n1;
                k2 = n1;
            else
                l2 = (b2-rem2)/n1 + 1;
                k2 = rem2;
            end
            if(l1 ~= l2)
                H(a,b) = 0;
            else
                tmp = y(l1,:).*y_pred(l1,:).*(1-y_pred(l1,:));
                tmp = tmp'.*(z(k1,:)').*(z(k2,:)');
                H(a,b) = sum(tmp);
            end
        end
        
        % one belonging to W1 and the other to W2
        if((a<=n1*n0) && (b>n1*n0))
            rem1 = mod(a,n0);
            if(rem1 == 0)
                k1 = a/n0;
                j1 = n0;
            else
                k1 = (a-rem1)/n0 + 1;
                j1 = rem1;
            end
            b2 = b - n1*n0;
            rem2 = mod(b2,n1);
            if(rem2 == 0)
                l2 = b2/n1;
                k2 = n1;
            else
                l2 = (b2-rem2)/n1 + 1;
                k2 = rem2;
            end
            tmp = y(l2,:).*y_pred(l2,:).*(1-y_pred(l2,:));
            tmp = tmp'.*(x(j1,:)').*(z(k2,:)').*(double(z(k1,:)>0)');
            %r1 = sum(tmp);
            %r2 = W2(l2,k1);
            H(a,b) = W2(l2,k1)*sum(tmp);
            if(k1 == k2)
                tmp2 = (y(l2,:)').*((1-y_pred(l2,:))').*(x(j1,:)').*(double(z(k2,:)>0)');
                %tmp2 = err(l2,:)'.*(x(j1,:)').*(double(z(k2,:)>0)');
                %H(a,b) = H(a,b) - sum(tmp2);
                H2(a,b) = -sum(tmp2);
            end
        end
        
        % symmetry
        H(b,a) = H(a,b);
        H2(b,a) = H2(a,b);
        
    end
    a
end

H = H/m;
H2 = H2/m;
eigvals = eig(H);
eigvals2 = eig(H2);
eigvals_sum = eig(H+H2);

%disp('No. of negative eigenvalues')
%neg_ct = length(eigvals(eigvals<0))
%disp('No. of positive eigenvalues')
%pos_ct = length(eigvals(eigvals>0))
%disp('Total no. of eigenvalues')
%neg_ct+pos_ct

%rank(H)
%rank(H+H2)
%max(eigvals2)

diff = eigvals_sum - eigvals;

H = H + H2;

%%% CCDF Plots code -
%x=linspace(-max(eigvals2),0,500);
%for i = 1:length(x)
%y(i) = length(diff(diff<x(i)))/length(diff(diff<0));end
%c=1/max(eigvals2);
%lambda_max = max(eigvals2);
%z1=(exp(15*c*(lambda_max+x))-1)/(exp(15*c*lambda_max)-1);
%z2=(exp(3*c*(lambda_max+x))-1)/(exp(3*c*lambda_max)-1);
%figure;plot(-x(x<=0),y(x<=0));hold;plot(-x(x<=0),z1(x<=0));plot(-x(x<=0),z2(x<=0),'g');xlabel('Absolute value of negative difference');ylabel('CCDF approximations');legend('Actual CCDF','CCDF Lower Bound','CCDF Upper Bound')

%eigvals(1:80)=0;
%sv=sqrt(eigvals);
%sv2=sv(81:7200);
%histfit(sv2-sv2(1),600,'exponential');
%fitdist(sv2-sv(1),'exponential')
%mu=0.33045;
%hist(diff(diff<0),200);
%sqrt(0.1)
%lambda_max=0.1;
%x=mu*sqrt(lambda_max);
%y=1+(2*(exp(-x)./x))-(2*((1-exp(-x))./(x.^2)));
%y
%y*7120

%histfit(-diff(diff<0),200,'exponential')
%fitdist(-diff(diff<0),'exponential')
%mu_prime=0.0191911;
%r=mu/sqrt(mu_p)
%r2=sqrt(mu_p*lambda_max)
%lambda_max
%alpha=(r/sqrt(2))*exp(r^2/4)*sqrt(2*pi)*(normcdf(r2*sqrt(2)+r/sqrt(2))-normcdf(r/sqrt(2)))
%alpha*1320

%p=zeros(800,1);
%p(1)=1;
%p(799)=1;
%p(800)=-1;
%r=roots(p);
%[a,b]=min(abs(imag(r)));
%r(b)
%-max(eigvals2)/(1699*log(r(b)))

%mu=0.2957*sigma*sqrt(y_pred*(1-y_pred))*(1+(params/m))
%mu=0.38*sigma*sqrt(y_pred*(1-y_pred))*(1+sqrt(params/m))
% k = 0.0871 or 0.09
%mu=1/mu
%lambda_max = 0.14*(1-exp(-0.7))
%p=zeros(750,1);
%p(1)=1;
%p(749)=1;
%p(750)=-1;
%r=roots(p);
%[a,b]=min(abs(imag(r)));
%r(b)
%mu_p=-lambda_max/(749*log(r(b)))
%mu_p=1/mu_p

%x = linspace(0,sqrt(lambda_max),1000);
%y = mu*exp(-(mu*x + (x.^4)/(2*lambda_max^2)));
%ar = trapz(x,y)

%x=linspace(-max(eigvals2),max(eigvals2),5000);
%for i = 1:length(x)
%y(i) = length(diff(diff<x(i)))/length(diff);
%end
%c=1/max(eigvals2);
%z=(exp(15*c*(lambda_max+x))-1)/(exp(15*c*lambda_max)-1);
%figure;plot(x(x<=0),y(x<=0));hold;plot(x(x<=0),0.3*z(x<=0))
