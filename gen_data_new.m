function [x,y] = gen_data_new()
n0 = 50;
n1 = 110;
n2 = 20;
m = 5500;
m_large = 25*m*n2;
x_large = randn(n0,m_large);
sigma =  1;
W1 = normrnd(0,sigma,[n1,n0]);
W2 = normrnd(0,sigma,[n2,n1]);
z = W1*x_large;
% ReLU 
z(z<0) = 0;
y0 = W2*z;
[~,y_large] = max(y0);
y_large = y_large';
%hist(y,200)

x = zeros(n0,m);
y = zeros(n2,m);
t = m/n2;
ct = 1;
for i = 1:n2
    pos = find(y_large == i);
    x(:,ct:ct+t-1) = x_large(:,pos(1:t));
    y(i,ct:ct+t-1) = 1;
    ct = ct+t;
end

x = x';
y = y';


