figure;
anti = abs(dihedrals - pi) < 0.2; 
gauche1 = abs(dihedrals - pi/6) < 0.1;
gauche2 = abs(dihedrals - 5*pi/6) < 0.1;
n_anti = sum(anti); 
n_gauche1 = sum(gauche1); 
n_gauche2 = sum(gauche2); 
scatter3(coords(:,1), coords(:,2), coords(:,3), ones(size(coords,1),1), 'blue'); 
hold on;
scatter3(coords(anti,1), coords(anti,2), coords(anti,3), ones(n_anti,1), 'red');
hold on;
scatter3(coords(gauche1,1), coords(gauche1,2), coords(gauche1,3), ones(n_gauche1,1), 'green');
hold on;
scatter3(coords(gauche2,1), coords(gauche2,2), coords(gauche2,3), ones(n_gauche2,1), 'black');

