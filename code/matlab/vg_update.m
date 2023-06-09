%% model
% x0:
% xp:
function [gradest, t] = vg_update(x0, xp, xq, wp, wq, xpt, xqt, sigma_list)

if(nargin < 8)
    sigma0 = comp_med(xq);
    sigma_list = linspace(sigma0*.2,sigma0*2,10);
end

% precompute distance for validation
distp = comp_dist([xpt, xqt], xp);
distq = comp_dist([xpt, xqt], xq);

npt = size(xpt,2);

t = [];
% model selection
parfor idx = 1:length(sigma_list)

    [~, r0] = grad_est([xpt, xqt], xp, xq, wp, wq, sigma_list(idx), distp, distq);
    if(isempty(r0))
        % if the grad est fail, continue to the next sigma
        t(idx) = inf;
        continue;
    end

    % testing score
    t(idx) = mean(-r0(1:npt) + log(mean(exp(r0(npt+1:end)))));
end

t

[~, idx] = min(t);
fprintf("sigma %f, index, %d\n", sigma_list(idx), idx);
% use the best sigma choice to perform grad est
gradest = grad_est(x0, xp, xq, wp, wq, sigma_list(idx));

end

%%
function [estgrad, r0] = grad_est(x0, xp, xq, wp, wq, sigma, distp, distq)

d = size(x0, 1);
estgrad = zeros(d, size(x0,2));
r0 = zeros(1, size(x0, 2));

% check if we need to compute distances between x0-xp and x0-xq
if(nargin > 6 && ~isempty(distp))
    kp = exp(- distp./sigma.^2/2);
else
    kp = exp(-comp_dist(x0, xp)./sigma.^2/2);
end

if(nargin > 7 && ~isempty(distq))
    kq = exp(- distq./sigma.^2/2);
else
    kq = exp(-comp_dist(x0, xq)./sigma.^2/2);
end

failed = false;
parfor i = 1:size(x0,2)
    % weighted grad est at x0
    Phi = [(xq - x0(:,i)); ones(1, size(xq,2))];
    Phi = Phi.*(kq(i,:).*wq)*Phi';
    psi = [(xp - x0(:,i)); ones(1, size(xp,2))];
    psi = psi*(kp(i,:).*wp)';

    % no need to remind us about ill conditioned least square
    ws = warning('off','all');

    alpha = Phi \ psi;
    if(isnan(sum(alpha)))
        % if the least square is ill-conditioned, set the flag
        failed = true;
    end


    alpha = alpha*kq(i,:)*wq'/(kp(i,:)*wp');
    estgrad(:, i) = alpha(1:end-1);

    r0(i) = alpha(end);

end

% least square is ill conditioned, return nothing
if(failed)
    estgrad = [];
    r0 = [];
end
end