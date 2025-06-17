function func = random_trigpoly(n, decay_func)
a = rand(1, n+1) / 2;
b = rand(1, n) / 2; 

for k = 2:n+1
    a(k) = a(k) * decay_func(k-1);
end
for k = 1:n
    b(k) = b(k) * decay_func(k);
end

func = chebfun(@(x) trig_poly_eval(x, a, b), [0, 1]);
figure;
plot(func);
title('Decayed Coefficients 10th Degree Trigonometric Polynomial');
xlabel('x');
ylabel('f(x)');
grid on;
end

function val = trig_poly_eval(x, a, b)
    val = a(1);
    n = length(b);
    for j = 1:n
        val = val + a(j+1) * cos(j * 2 * pi * x) + b(j) * sin(j * 2 * pi * x);
    end
end
