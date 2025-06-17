N = 4000;
gamma = 2.5;
tau = 7;
sigma = 49;

visc = 1e-3;

s = 1024;
T = 1;
steps = 100 * T;

input = zeros(N, s);
params = zeros(N, 3);
if steps == 1
    output = zeros(N, s);
else
    output = zeros(N, steps, s);
end

ubase = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
ubase2 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
%ubase3 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
%ubase4 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");

[a1, b1] = get_trigcoeffs(ubase);
[a2, b2] = get_trigcoeffs(ubase2);
%[a3, b3] = get_trigcoeffs(ubase3);
%[a4, b4] = get_trigcoeffs(ubase4);

tspan = linspace(0,T,steps+1);
x = linspace(0,1,s+1);
for j=1:N
    tic
    
    a = rand + 0.5;
    b = rand + 0.5;
    %c = rand;
    %d = rand;
    h = rand;
    u0 = chebfun(@(x) 1/2 * (a * ubase(x - h) + b * ubase2(x - h)), [0 1]);
    u = burgers1(u0, tspan, s, visc, j==1);
    
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
    params(j, 3) = h;
    %params(j, 4) = d;
    %params(j, 5) = h;
    
    disp(j); 
    toc
end

alldata = output(:, 1:2:101, :);
save("grfarc2visc0p001-threeparamtry2.mat", "alldata", "params", "a1", "b1", "a2", "b2");

function u = burgers1(init, tspan, s, visc, plot)

S = spinop([0 1], tspan);
dt = tspan(2) - tspan(1);
S.lin = @(u) + visc*diff(u,2);
S.nonlin = @(u) - 0.5*diff(u.^2);
S.init = init;

if plot
    u = spin(S,s,dt); 
else
    u = spin(S,s,dt, 'plot','off'); 
end

end

function u = GRF1(N, m, gamma, tau, sigma, type)

if type == "dirichlet"
    m = 0;
end

if type == "periodic"
    my_const = 2*pi;
else
    my_const = pi;
end

my_eigs = sqrt(2)*(abs(sigma).*((my_const.*(1:N)').^2 + tau^2).^(-gamma/2));

if type == "dirichlet"
    alpha = zeros(N,1);
else
    xi_alpha = randn(N,1);
    alpha = my_eigs.*xi_alpha;
end

if type == "neumann"
    beta = zeros(N,1);
else
    xi_beta = randn(N,1);
    beta = my_eigs.*xi_beta;
end

a = alpha/2;
b = -beta/2;

c = [flipud(a) - flipud(b).*1i;m + 0*1i;a + b.*1i];

if type == "periodic"
    uu = chebfun(c, [0 1], 'trig', 'coeffs');
    u = chebfun(@(t) uu(t - 0.5), [0 1], 'trig');
else
    uu = chebfun(c, [-pi pi], 'trig', 'coeffs');
    u = chebfun(@(t) uu(pi*t), [0 1]);
end
end

function [a, b] = get_trigcoeffs(func)
C = trigcoeffs(func);

N = floor(length(C)/2);
a = zeros(1, N+1);
b = zeros(1, N);
a(1, N+1) = real(C(N+1));

for k = 1:N
    a(k) = real(C(N+1+k) + C(N+1-k));
    b(k) = imag(C(N+1+k) - C(N+1-k));
end

end
