%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function tries to solve Euler's rigid body equation
% I is the moment of inertia, col
% Y0 is the initial condition, col
% h is time step
% n is number of iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Y] = Euler_rigid_ani(I, h, n)

num_curve = 20; % roughly how much contour curves on sphere, due to symmetry, will be less
p_ini = 14; % moving point on which contour curve
Y = zeros(3,n,num_curve);


theta = linspace(-pi+0.1, pi-0.1 ,num_curve);
for j = 1:num_curve
    Y(:,1,j) = [cos(theta(j)); 0 ; sin(theta(j))]; 
end
a = zeros(3,1);
prodI = sqrt(I(1)*I(2)*I(3));
a(1) = (I(2)-I(3))/prodI;
a(2) = (I(3)-I(1))/prodI;
a(3) = (I(1)-I(2))/prodI;

opts = optimset('Algorithm','trust-region-reflective','Diagnostics','off', 'Display','off');

v = VideoWriter('Euler_rigid.mpeg','MPEG-4');
% v.CompressionRatio = 3;
v.Quality = 100;
open(v);

figure()
% u = uicontrol('Style','slider','Position',[10 50 20 340],...
%     'Min', 1, 'Max', n, 'Value', 1);
[x,y,z] = sphere;
F=surfl(sqrt(1)*x,sqrt(1)*y,sqrt(1)*z);
set(F, 'FaceAlpha', 0.6)
shading interp
colormap(gray)
view([145 25])

for j = 1:num_curve
    for i =2 : n
       Yp = Y(:,i-1,j) + h*[a(1)*Y(2,i-1,j)*Y(3,i-1,j);...
           a(2)*Y(1,i-1,j)*Y(3,i-1,j);...
           a(3)*Y(1,i-1,j)*Y(2,i-1,j)]; 

       Y(:,i,j) = fsolve(@(x) E_CI(Y(:,i-1,j), x), Yp, opts);

    end
end
hold on
for j = 1:num_curve
    plot3(Y(1,:,j)',Y(2,:,j)',Y(3,:,j)','k');
end

quiver3(0,0,0,1.2,0,0,'k--','LineWidth',0.8); text(1.1,0,0,'X');
quiver3(0,0,0,0,1.2,0,'k--','LineWidth',0.8); text(0,1.1,0,'Y');
quiver3(0,0,0,0,0,1.2,'k--','LineWidth',0.8); text(0,0,1.1,'Z');
scatter3(1,0,0,'k','filled');
scatter3(0,1,0,'k','filled');
scatter3(0,0,1,'k','filled');

p = scatter3(Y(1,1,p_ini),Y(2,1,p_ini),Y(3,1,p_ini),40,'filled');
axis ([-1 1 -1 1 -1 1])
for i =2 : n  
   p.XData = Y(1,i,p_ini);
   p.YData = Y(2,i,p_ini);
   p.ZData = Y(3,i,p_ini);
%    pause(.1)     %uncomment it if you want animation be slower
   writeVideo(v,getframe);
   drawnow
end


close(v)
hold off

% box off

% figure()
% plot(E)
% figure()
% plot(M)

    function y = E_CI(xc, x)
        Jn = x(1)*x(2)*x(3);
        J = xc(1)*xc(2)*xc(3);
        y = [x(1)^2 - xc(1)^2 - h*a(1)*(J+Jn);...
            x(2)^2 - xc(2)^2 - h*a(2)*(J+Jn);...
            x(3)^2 - xc(3)^2 - h*a(3)*(J+Jn)];
    end
end
