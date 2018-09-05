function [eigvals,eigvals2,eigvals_sum] = sim_Hessian4(x_init,W1_init,W2_init,y,y_pred)
% W1 is n1*n0, W2 is 1*n1
% m = no. of training examples
% x,W1,W2 are assumed to be normally distributed
%n0 = 51;
%n1 = 101;
n0 = 50;
n1 = 110;
n2 = 20;
m = 5500;

%W1 = zeros(n1,n0);
%W1(1:n1-1,:) = W1_init';
%W1(n1,n0) = 1;
%W2 = W2_init';
%x = ones(n0,m);
%x(1:n0-1,:) = x_init';
%[eigvals,eigvals11,eigvals22] = sim_Hessian4(x_train_aug,W1_aug,W2_aug,y_train',Y_pred');
W1 = W1_init';W2=W2_init';x=x_init';

z = W1*x;
% ReLU 
z(z<0) = 0;

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

%W1_aug = zeros(101,41);
%W1_aug(1:100,:) = W1';
%W1_aug(101,41) = 1;
%W2_aug = W2';
%x_train_aug = ones(41,4500);
%x_train_aug(1:40,:) = x_train';
%[eigvals,eigvals11,eigvals22] = sim_Hessian4(x_train_aug,W1_aug,W2_aug,y_train',Y_pred');

length(eigvals_sum(eigvals_sum<0))/params

