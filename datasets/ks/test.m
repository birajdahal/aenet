tic, dom = [0 50]; x = chebfun('x',dom); tspan = [0 10];
S = spinop(dom,tspan);
S.lin = @(u) - diff(u,2) - diff(u,4);
S.nonlin = @(u) -.5*diff(u.^2); % spin cannot parse "u.*diff(u)"

N = 5;

us = [];
for i = 1:N
a = 10*rand;
b = rand;
S.init = (1+3*b) * exp(-((x-25-a)/5).^2);
tic, u = spin(S,1024,0.1, 'plot', 'off');

us = [us, u];
end

for i = 1:N
    plot(us(:, i))
    hold on
end
hold off