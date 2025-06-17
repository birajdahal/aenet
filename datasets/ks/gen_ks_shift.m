N = 1000;

s = 1024;
T = 10;
steps = 10 * T;
dom = [0, 50];

input = zeros(N, s);
params = zeros(N, 2);
if steps == 1
    output = zeros(N, s);
else
    output = zeros(N, steps, s);
end

tspan = linspace(0,T,steps+1);
x = linspace(dom(1),dom(2),s+1);
for j=1:N
    tic
    
    a = rand * 20;
    b = 0;
    u0 = chebfun(@(x) (1+3*b) * exp(-((x-10-a)/5).^2), dom);
 
    u = ks1(u0, dom, tspan, s, j==1);
    
    u0eval = u0(x);
    input(j,:) = u0eval(1:end-1);
    
    if steps == 1
        output(j,:) = u.values;
    else
        output(j, 1, :) = input(j, :);
        for k=2:(steps+1)
            output(j,k,:) = u{k}.values;
        end
    end
    
    params(j, 1) = a;
    params(j, 2) = b;
    
    disp(j);
    toc
end

%ksshift = output(:, 1:5:101, :);
save("ksshiftall.mat", "output", "params");

function u = ks1(init, dom, tspan, s, plot)
S = spinop(dom, tspan);
dt = tspan(2) - tspan(1);
S.lin = @(u) - diff(u,2) - diff(u,4);
S.nonlin = @(u) -.5*diff(u.^2); % spin cannot parse "u.*diff(u)"
S.init = init;

if plot
    u = spin(S,s,dt); 
else
    u = spin(S,s,dt, 'plot','off'); 
end

end