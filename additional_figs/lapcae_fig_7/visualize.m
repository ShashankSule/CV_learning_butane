close all
dih = readmatrix("oct2_dihedrals_mashameeting.csv");
coord = readmatrix("oct2_pointcloud_mashameeting.csv");

figure(1); clf;
scatter3(coord(:,1),coord(:,2),coord(:,3),5*ones(size(dih)),dih,'filled')
colormap hsv
daspect([1,1,1])
alpha(0.1)
% Create a translucent plane through the xy-plane
hold on;
[xPlane, yPlane] = meshgrid(-0.2:0.04:0.2, -0.2:0.04:0.2); % Plane grid points
zPlane = zeros(size(xPlane)); % z = 0 for the xy-plane
plane = surf(xPlane, yPlane, zPlane); % Create the surface
set(plane, 'FaceColor', 'black', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
xlabel('$\psi_1$', 'Interpreter','latex', 'FontSize',24)
ylabel('$\psi_2$', 'Interpreter','latex', 'FontSize',24)
zlabel('$\psi_3$', 'Interpreter','latex', 'FontSize',24)
c = colorbar;
c.Label.String = 'Dihedral angle';
c.Label.FontSize = 24;

% figure(2); clf;
% scatter3(coord(:,1),coord(:,3),coord(:,4),2*ones(size(dih)),'filled')
% colormap hsv
% daspect([1,1,1])
% alpha(0.1)
% xlabel('$\psi_1$', 'Interpreter','latex', 'FontSize',24)
% ylabel('$\psi_3$', 'Interpreter','latex', 'FontSize',24)
% zlabel('$\psi_4$', 'Interpreter','latex', 'FontSize',24)
% 
% figure(3); clf;
% scatter3(coord(:,1),coord(:,2),coord(:,3),2*ones(size(dih)),'filled')
% colormap hsv
% daspect([1,1,1])
% fontsize(24,"points")

% % slices
% min1 = min(coord(:,1));
% max1 = max(coord(:,1));
% n = 12;
% ni = 4;
% nj = n/ni;
% c1 = linspace(min1,max1,n+1);
% figure(4); 
% for i = 1 : ni
%     for j = 1 : nj
%         k = j + (i-1)*nj;
%         ind = find(coord(:,1) >= c1(k) & coord(:,1) < c1(k+1));
%         subplot(ni,nj,k);
%         scatter(coord(ind,3),coord(ind,4),2*ones(size(dih(ind))),dih(ind))
%         axis([-0.2,0.2,-0.2,0.2])
%         xlabel('$\psi_1$', 'Interpreter','latex', 'FontSize',16)
%         ylabel('$\psi_2$', 'Interpreter','latex', 'FontSize',16)
%         colormap hsv
%     end
% end
% 
% % slices
% min1 = min(coord(:,1));
% max1 = max(coord(:,1));
% n = 12;
% ni = 4;
% nj = n/ni;
% c1 = linspace(min1,max1,n+1);
% figure(5); 
% for i = 1 : ni
%     for j = 1 : nj
%         k = j + (i-1)*nj;
%         ind = find(coord(:,1) >= c1(k) & coord(:,1) < c1(k+1));
%         subplot(ni,nj,k);
%         scatter3(coord(ind,2),coord(ind,3),coord(ind,4),2*ones(size(dih(ind))),dih(ind))
%         axis([-0.2,0.2,-0.2,0.2,-0.2,0.2])
%         xlabel('$\psi_1$', 'Interpreter','latex', 'FontSize',16)
%         ylabel('$\psi_2$', 'Interpreter','latex', 'FontSize',16)
%         zlabel('$\psi_1$', 'Interpreter','latex', 'FontSize',16)
%         colormap hsv
%     end
% end

