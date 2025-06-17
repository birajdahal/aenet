% number of realizations to generate
N = 2500;

% grid size
T = 0.01; 
s = 2048;  
steps = 300;

input = zeros(N, s);
if steps == 1
    output = zeros(N, s);
else
    output = zeros(N, steps, s);
end

params = zeros(N, 2);
tspan = linspace(0,T,steps+1);
x = linspace(0,1,s+1);
for j=1:N
    tic

    A = 16;
    [u0, a, h] = solitons();
    u = kdv1(u0, [0, 6], tspan, s, 0);
    params(j, 1) = a;
    params(j, 2) = h;
    
    u0eval = u0(x);
    
    if steps == 1
        output(j,:) = u.values;
    else
        output(j, 1, :) = u{1}.values;
        for k=2:(steps+1)
            output(j,k,:) = u{k}.values;
        end
    end
    
    disp(j);
    toc
end

kdv2wide = output(:, 1:6:301, :);
save("kdv2-aenet.mat", "kdv2wide", "params");

function [u, a, h] = solitons()
    a = rand * 12 + 6; 
    h = rand * 3;
    u = chebfun(@(x) 0.5*a^2*sech(0.5*a*(x-1))^2 + 0.5*6^2*sech(0.5*6*(x-2-h))^2, [0, 6]);
end

function u = kdv1(init, dom, tspan, s, plot)

S = spinop(dom, tspan);
dt = tspan(2) - tspan(1);
S.lin = @(u) -1*diff(u,3);
S.nonlin = @(u) -3*diff(u.^2);
S.init = init;

if plot
    u = spin(S,s,dt); 
else
    u = spin(S,s,dt, 'plot','off'); 
end

end


