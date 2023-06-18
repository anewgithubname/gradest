clear
addpath("matlab");

%%
% problem set up
seed = 1
sigmap = 5
mup = 0
rng(seed)
n = 100; % number of particles
d= 2;
%%
% sample from the target 
xp = [randn(d,n)*.25+2];

% grad log p
logp = @(x) log(mvnpdf(x', zeros(1,d)+2, eye(d)/16))';
x = sym('x', [d,1], 'real'); 
nabla_logp = matlabFunction(gradient(logp(x)),'var',{x});

% initial particles, from N(0,1)
xq = [randn(d,n)];

h = figure; 
for iter = 1:50

    [estgrad] = svgd(xq, nabla_logp, false); 
    xq = xq + .01*estgrad;

    clf;
    hold on; 
    scatter(xp(1,:), xp(2,:), '.', 'DisplayName', 'target dist.');
    scatter(xq(1,:), xq(2,:), '.', 'r', 'DisplayName', 'particle dist.');

    title(sprintf("SVGD WITHOUT normalization, iteration %d", iter));

    axis([-2,4,-2,4]); 
    drawnow;
end

title("Press any key to run SVGD WITH normalization...")
drawnow

pause

xq = [randn(d,n)];
for iter = 1:50

    [estgrad] = svgd(xq, nabla_logp, true); 
    xq = xq + .01*estgrad;

    clf;
    hold on; 
    scatter(xp(1,:), xp(2,:), '.', 'DisplayName', 'target dist.');
    scatter(xq(1,:), xq(2,:), '.', 'r', 'DisplayName', 'particle dist.');
    
    title(sprintf("SVGD WITH normalization, iteration %d", iter));
    axis([-2,4,-2,4]); 
    drawnow;
end

%% SVGD update

function [grad] = svgd(X, gradlogp, normalize)

% transpose, so that rows are obs and cols are features
X = X';

% median trick. You can play with this, median /2, median *2 etc. 
sigma = comp_med(X');

% compute kernel matrix
Xb = X; % pick your basis. The classic SVGD uses all particles as basis
D = comp_dist(X', Xb');
n = size(X,1);
nb = size(Xb,1);
K = kernel_gau(D, sigma); % K is n \times nb

% evaluate gradient for each Xb
glogp = [];
for i = 1:size(Xb,1)
    glogp(:, i) = gradlogp(Xb(i,:)');
end

grad = zeros(n, size(X,2));
for i = 1:size(X,2) 
    logp_K = repmat(glogp(i,:), n, 1) .* K;

    diff = (repmat(X(:,i),1,nb) - repmat(Xb(:,i)',n,1))/sigma^2;

    % SVGD = E_xb(nabla_xb logp K + nabla_xb K)
    grad(:, i) = mean(logp_K + diff.*K, 2);

    % normalize SVGD or not? 
    if normalize
        grad(:, i) = grad(:, i)./(sum(K,2)/n);
    end
end

% transpose back to row feature, col observation
grad = grad';
end
