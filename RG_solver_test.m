%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This script compares different solver for mathematical pendulum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [qs,qH,q2] = RG_solver_test()

n = 30;  % number of simulated points
dt = .2; % time step

H_func=@(p, q) 0.5*p^2 -cos(q);

x = linspace(-pi, pi);
y = linspace(-2,2);
[X, Y] = meshgrid(x, y);
Z = 0.5*Y.^2-cos(X);
% figure()
% contour(X,Y,Z)


p = zeros(n,1);
q = zeros(n,1);
H = zeros(n,1);

%% Explicit Euler
p(1) = 1;
q(1) = 0;
H(1) = H_func(p(1), q(1));
for i = 1 : n-1
    q(i+1) = q(i) + dt*p(i);
    p(i+1) = p(i) - dt*sin(q(i));
    H(i+1) = H_func(p(i+1),q(i+1));
end
    
figure
hold on
contour(X,Y,Z)
plot(q,p,'o');
hold off
title('phase trajactory of Euler forward')
xlabel('q')
ylabel('p')
figure
plot(H)
ylim([-1 2])
title('change of H using Euler forward')

%% Simpletic Euler
qs = q;
ps = p;
Hs = H;
for i = 1 : n-1
    qs(i+1) = qs(i) + dt*ps(i);
    ps(i+1) = ps(i) - dt*sin(qs(i+1));
    Hs(i+1) = H_func(ps(i+1),qs(i+1));
end
    
figure
hold on
contour(X,Y,Z)
plot(qs,ps,'o');
hold off
title('phase trajactory of simpletic Euler')
xlabel('q')
ylabel('p')
figure
plot(Hs)
ylim([-1 2])
title('change of H using simpletic Euler')

%% Euler backward
q1 = q;
p1 = p;
H1 = H;
opts = optimset('Algorithm','trust-region-reflective','Diagnostics','off', 'Display','off');

for i = 2: n
    sol= fsolve(@(x)E_back([p1(i-1); q1(i-1)], x), [p1(i-1); q1(i-1)],opts);
    p1(i) = sol(1);
    q1(i) = sol(2);
    H1(i) = H_func(p1(i),q1(i));
end

figure
hold on
contour(X,Y,Z)
plot(q1,p1,'o');
hold off
title('phase trajactory of Euler backward')
xlabel('q')
ylabel('p')
figure
plot(H1)
ylim([-1 1])
title('change of H using Euler backward')


%% Euler in q, solve p from H
qH = q;
pH = p;
HH = H;
for i = 1 : n-1
    if abs(pH(i))> 0.1
        qH(i+1) = qH(i) + dt*pH(i);
        pH(i+1) = fsolve(@(x) 0.5*x^2-cos(qH(i+1))-H1(1), pH(i),opts);
    else
        pH(i+1) = pH(i) + dt*(-sin(qH(i)));
        qH(i+1) = fsolve(@(x) 0.5*pH(i+1)^2-cos(x)-H1(1), qH(i),opts);
    end
    HH(i+1) = H_func(pH(i+1),qH(i+1));
end
    
figure
hold on
contour(X,Y,Z)
plot(qH,pH,'o');
hold off
title('phase trajactory')
xlabel('q')
ylabel('p')
figure
plot(HH)
ylim([-1 2])
title('change of H')



%% Heun
q2 = q;
p2 = p;
H2 = H;

for i = 2: n
    sol= fsolve(@(x)E_half([p2(i-1); q2(i-1)], x), [p2(i-1); q2(i-1)],opts);
    p2(i) = sol(1);
    q2(i) = sol(2);
    H2(i) = H_func(p2(i),q2(i));
end

figure
hold on
contour(X,Y,Z)
plot(q2,p2,'o');
hold off
title('phase trajactory of Heun method')
xlabel('q')
ylabel('p')
figure
plot(H2)
ylim([-1 1])
title('change of H using Heun method')

%% RK(4, 5) adaptive
options = odeset('RelTol',1e-6,'AbsTol',[1e-6;1e-6]);
[~,Sol] = ode45(@f_func,[-pi,pi],[p(1);q(1)],options);
figure
hold on
contour(X,Y,Z)
plot(Sol(:,1),Sol(:,2),'o');
hold off
title('phase trajactory of RK')
xlabel('q')
ylabel('p')
figure
plot(Sol(:,1).^2*0.5-cos(Sol(:,2)));
xlim([0 100])
ylim([-1 1])
title('change of H using RK')


function y = E_back(xc, xn)
    y = [xn(1) - xc(1)+dt*sin(xn(2)); xn(2) - xc(2)-dt*xn(1)];
    
end

function y = E_half(xc, xn)
    y = [xn(1) - xc(1)+0.5*dt*(sin(xn(2))+sin(xc(2))); xn(2) - xc(2)-0.5*dt*(xn(1)+xc(1))];
    
end

    function y = f_func(t,x)
        y = [-sin(x(2)); x(1)];
    end

end