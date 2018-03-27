clear all

Images = loadMNISTImages('./MNIST/t10k-images.idx3-ubyte');
Images = reshape(Images, 28,28,[]);
Labels = loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10; % 0 --> 10

rng(1);

% Learning
%
W1 = 1e-2*randn([9 9 20]);
W5 = (2*rand(100,2000) - 1) * sqrt(6) / sqrt(360 + 2000);
Wo = (2*rand( 10, 100) - 1) * sqrt(6) / sqrt( 10 +  100);

X = Images(:, :, 1:8000);
D = Labels(1:8000);

for epoch = 1:5
    epoch
    [W1, W5, Wo] = MnistConv(W1, W5, Wo,X,D);
end

save('MnistConv.mat');

%% Test
%
X = Images(:,:,8001:10000);
D = Labels(8001:10000);

acc = 0;
N = length(D);
for k = 1:N
   x = X(:,:,k);
   y1 = Conv(x,W1);
   y2 = ReLU(y1);
   y3 = Pool(y2);
   y4 = reshape(y3,[],1);
   v5 = W5*y4;
   y5 = ReLU(v5);
   v = Wo*y5;
   y = Softmax(v);
   
   [~,i] = max(y);
   if i == D(k)
       acc = acc + 1;
   end
end

acc = acc / N;
fprintf('Accuracy is %f\n', acc);

function y = Conv(x,W)
%
%
    [wrow, wcol, numFilters] = size(W);
    [xrow, xcol, ~        ] = size(x);
    
    yrow = xrow - wrow + 1;
    ycol = xcol - wcol + 1;
    
    y = zeros(yrow, ycol, numFilters);
    
    for k = 1:numFilters
        filter = W(:,:,k);
        filter = rot90(squeeze(filter),2);
        y(:,:,k) = conv2(x, filter, 'valid');
    end
end

function y = Pool(x)
%
% 2x2 mean pooling
%
[xrow, xcol, numFilters] = size(x);

y = zeros(xrow/2, xcol/2, numFilters);
    for k = 1:numFilters
        filter = ones(2) / (2*2); % for mean
        image = conv2(x(:,:,k), filter, 'valid');
        y(:,:,k) = image(1:2:end, 1:2:end);
    end
end

function y = ReLU(x)
    y = max(0,x);
end

function y = Softmax(x)
    ex = exp(x);
    y = ex / sum(ex);
end
