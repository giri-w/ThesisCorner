clear all

X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      0
      1
      0.7
    ];

W = 2*rand(1,3) - 1;

for epoch = 1:10000
   W = DeltaSGD(W, X, D); 
end

N = 4;
for k = 1:N
   x = X(k,:)';
   v = W*x;
   y = Sigmoid(v)
end

function y = Sigmoid(x)
    y = 1/ (1 + exp(-x));
end