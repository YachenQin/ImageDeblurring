
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% prepare workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; home;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load high-resolution image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load a standard MATLAB image Cameraman
x = imread('cameraman.tif');
x = im2double(x);

% size of the image
imgSize = size(x);

% create a figure and show the image
figure('Color', 'w', 'Name', 'x');
imagesc(x);
axis equal off;
colormap gray; % show as black and white

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define blurring operator and its adjoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define a blurring kernel
h = ones(3)/3;

% blur in Fourier domain
H = fft2(h, imgSize(1), imgSize(2));

% define operators
amult = @(x) real(ifft2(fft2(x).*H));
atran = @(x) real(ifft2(fft2(x).*conj(H)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate a low-res image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y = amult(x);

% create a figure and show the image
figure('Color', 'w', 'Name', 'y');
imagesc(y);
axis equal off;
colormap gray; % show as black and white
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% super-resolution parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xhat0 = y; % initial value
gamma = 0.01; % step-size
tol = 1e-6; % termination tolerance for gradient
maxiter = 1000; % maximum number of iterations to run

% cost function
f = @(x) 0.5*sum(sum(abs(y-amult(x)).^2));
grad = @(x) atran(amult(x)-y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% optimize: gradient method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize
xhat = xhat0;

% track the evolution of the objective
fvals = zeros(maxiter, 1);

for iter = 1:maxiter
    
    % compute the next iterate
    xhatnext = xhat - gamma*grad(xhat);
    
    % store the objective value
    fvals(iter) = f(xhatnext);
    
    % plot the results
    figure(101);
    set(gcf, 'Color', 'w', 'Name', sprintf('Obj = %.2e', fvals(iter)));
    
    subplot(3, 2, 1:4)
    imagesc(xhat);
    axis equal off;
    colormap gray; % show as black and white
    
    subplot(3, 2, 5:6)
    semilogy(1:iter, fvals(1:iter), 'b-',...
        iter, fvals(iter), 'r.', 'LineWidth', 2);
    grid on;
    xlabel('iteration');
    ylabel('objective');
    xlim([1 maxiter]);
    set(gca, 'FontSize', 16);
    drawnow;
    
    % check termination tolerance
    if(sqrt(sum(sum(grad(xhatnext).^2))) < tol)
        break;
    end
    
    % update the iterate
    xhat = xhatnext;
end

xhatgm = xhat;
