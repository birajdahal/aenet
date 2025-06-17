a = a1;
b = b1;

u_t = a(512); 
t = linspace(0, 1);

for k = 1:length(b)
    u_t = u_t + a(k) * cos(2*pi*k*(1-t)) + b(k) * sin(2*pi*k*(1-t));
end
