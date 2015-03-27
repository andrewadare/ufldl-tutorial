## Utility/common functions for deep learning tutorial
using NLopt

# Logistic sigmoid function and its gradient
sigmoid(z) = 1 ./ (1 + exp(-z))
sigmoidGradient(z) = sigmoid(z).*(1 - sigmoid(z))

# Kullback-Leibler divergence between Bernoulli variables with means p and q.
klBernoulli(p,q) = sum(p.*log(p./q) + (1-p).*log((1-p)./(1-q)))

# Return image patches in the columns of a (patchsize^2 x npatches) matrix.
function sampleImages(imgs; patchsize = 8, npatches = 10000, normalize = true)

    # Sample random patches from larger images
    patches = zeros(patchsize*patchsize, npatches)
    r,c,m = size(imgs)
    for i = 1:npatches
        # Pick a random upper left corner and get the patch
        x = rand(1:c-patchsize+1)
        y = rand(1:r-patchsize+1)
        patch = imgs[x:x+patchsize-1,y:y+patchsize-1,rand(1:m)]
        patches[:,i] = patch[:]
    end

    if normalize
        # Center patches
        for j = 1:npatches
            patches[:,j] -= mean(patches[:,j])
        end

        # Truncate to +/-3 standard deviations and scale to -1 to 1
        d = 3*std(patches)
        patches = max(min(patches, d), -d) / d

        # Rescale from [-1,1] to [0.1,0.9]
        patches = (patches + 1) * 0.4 + 0.1
    end
    patches
end

# Return a square mosaic of N² images selected from an m×n dataset X
# where m is the number of images and each image is √n x √n.
function imageMosaic(X, N; random = true)
    m,n = size(X)
    h,w = int(sqrt(n)), int(sqrt(n))
    mosaic = zeros(Float64, N*h, N*w)
    entry = 0
    for i in 1:N
        for j in 1:N
            entry = random ? rand(1:m) : entry + 1
            digit = reshape(X[entry,:], h, w)'
            i1 = h*i - h + 1 
            i2 = i1 + h - 1
            j1 = w*j - w + 1
            j2 = j1 + w - 1
            mosaic[i1:i2, j1:j2] = digit
        end
    end
    mosaic
end

# Randomly initialize the two weight matrices for a 3-layer autoencoder network 
# with nv nodes in the visible layers and nh nodes in the hidden layer.
function initWeights(nv, nh)
    # Random (uniform) interval is [-r,r]. 
    # This is the range chosen by the tutorial instructors.
    r = sqrt(6) / sqrt(nv+nh+1)
    W1 = rand(nh, nv)*2*r - r
    W2 = rand(nv, nh)*2*r - r
    b1 = zeros(nh,1)
    b2 = zeros(nv,1)

    # Unroll the parameters into a single vector (size nh*nv*2 + nh + nv)
    # suitable for input to optimizer.
    theta = [W1[:]; W2[:]; b1[:]; b2[:]]
end

# Sparse autoencoder cost function and gradient with respect to theta
function saeCost(theta, nv, nh, lambda, beta, rho, data)
    # Θ:        AE weights and biases from initWeights() (one big vector).
    # nv, nh:   number of visible and hidden units (probably 64 and 25).
    # λ:        Weight decay parameter
    # β:        Weight of sparsity penalty term
    # ρ:        Sparsity param. Desired average activation for the hidden units.
    # data:     64x10000 matrix containing the training data (1e4 8x8 patches).  
    #           So, data[:,i] is the i-th training example. 
  
    # Get original components of Θ from the rolled-up vector.
    W1 = reshape(theta[1:nh*nv], nh, nv)
    W2 = reshape(theta[nh*nv+1:2*nh*nv], nv, nh)
    b1 = theta[2*nh*nv+1:2*nh*nv+nh]
    b2 = theta[2*nh*nv+nh+1:end]

    gradW1, gradW2 = zeros(W1), zeros(W2) # nh x nv, nv x nh
    gradb1, gradb2 = zeros(b1), zeros(b2) # nh, nv

    # Forward propagation ------------------------------------------------------
    # Compute ||h - y||² via FP where h = a3. (See eq 6-8 in notes).
    # Since an AE is an identity approximator, y = data.
    # Run FP on all 10000 data examples at once (hence the repmat; m is the
    # number of data examples). a2 and a3 are stored for backpropagation.
    m = size(data, 2)
    z2 = W1*data + repmat(b1, 1, m)
    a2 = sigmoid(z2)
    z3 = W2*a2 + repmat(b2, 1, m)
    a3 = sigmoid(z3)

    # Reproduction error of autoencoder and squared-error contribution
    e = a3 - data
    seCost = 0.5/m * sum(e.^2)

    # Regularization term
    regCost = 0.5*lambda*sum(W1[:].^2 + W2[:].^2)

    # Sparsity penalty term. 
    # rhoHat is a vector of mean activations of a2 over the data.
    rhoHat = 1/m * sum(a2, 2)
    spCost = beta*klBernoulli(rho, rhoHat)

    J = seCost + regCost + spCost
    # println("SE, reg, sp: $seCost, $regCost, $spCost")

    # Backpropagation ----------------------------------------------------------
    # Sparsity penalty to be added to delta2. 
    # This is a vector; repeat m times when computing delta2.
    kld = (1-rho)./(1-rhoHat) - (rho./rhoHat)

    delta3 = e .* (a3 .* (1 - a3))
    delta2 = (W2'*delta3 + beta*repmat(kld,1,m)) .* (a2 .* (1 - a2))

    # Compute gradients --------------------------------------------------------
    gradW2 = delta3*a2'/m + lambda*W2
    gradW1 = delta2*data'/m + lambda*W1
    gradb2 = sum(delta3,2)/m
    gradb1 = sum(delta2,2)/m

    J, [gradW1[:]; gradW2[:]; gradb1[:]; gradb2[:]]
end

function numericalGradient(J, theta)
    # J: Cost function (the function itself--not the return value)
    # Θ: AE weights and biases from initWeights() (one big vector).
    numgrad = zeros(theta)
    eps = 1e-4
    n = size(numgrad,1)
    for i = 1:n
        epsvec = zeros(n)
        epsvec[i] = eps
        a, grada = J(theta - epsvec)
        b, gradb = J(theta + epsvec)
        numgrad[i] = (b-a) / (2*eps)
    end
    numgrad
end

function checkGradient()
    quadTest(x) = (x[1]^2 + 3x[1]*x[2], [2x[1]+3x[2], 3x[1]])
    x = [4.0; 10.0]
    val, grad = quadTest(x)
    numgrad = numericalGradient(quadTest, x)
    display([numgrad grad])
    diff = norm(numgrad-grad)/norm(numgrad+grad)
    report = diff < 1e-9 ? "PASS" : "FAIL"
    println("$report: Numerical - analytic gradient norm difference = $diff.")
end

# Reshape columns of A into a square matrix and tile them into a square grid.
function tileColumns(A::Matrix)
    m,n = size(A)
    l = int(sqrt(m))
    ncols = int(sqrt(n))
    mosaic = zeros(Float64, ncols*l, ncols*l)

    for j in 1:n
        Aj = A[:,j]
        Aj -= mean(A)
        Aj /= maximum(abs(Aj))

        row = int(ceil(j/ncols))
        col = j % ncols + (j % ncols == 0 ? ncols : 0)

        a = (row-1)*l + 1
        b = a + l - 1
        c = (col-1)*l + 1
        d = c + l - 1 
        mosaic[a:b, c:d] = reshape(Aj, l, l)
        # println("j=$j, row=$row, col=$col [$a:$b, $c:$d]")
    end
    mosaic
end

function softmaxCost(theta, ncat, data, labels, lambda)
    n,m = size(data) # n is input size, m is number of entries. 
    theta = reshape(theta, ncat, n)

    # Create ground-truth matrix M such that M[r[i],c[i]] = 1 (size ncat x m).
    # This is the matrix of indicator functions 1{y=j}.
    ind = full(sparse(int(labels), [1:m;], 1))

    ## For MNIST, the first 10 columns of y look like this (row 10 is the 0 digit):
    #
    #    0  0  0  1  0  0  1  0  1  0
    #    0  0  0  0  0  1  0  0  0  0
    #    0  0  0  0  0  0  0  1  0  0
    #    0  0  1  0  0  0  0  0  0  1
    #    1  0  0  0  0  0  0  0  0  0
    #    0  0  0  0  0  0  0  0  0  0
    #    0  0  0  0  0  0  0  0  0  0
    #    0  0  0  0  0  0  0  0  0  0
    #    0  0  0  0  1  0  0  0  0  0
    #    0  1  0  0  0  0  0  0  0  0

    # Compute exp(theta*x).
    # Handle numerical overflow by subtracting off the largest value in each
    # column of tx, as explained in the tutorial. This columnwise subtraction
    # cancels in the ratio later, so this does not affect the final answer.
    tx = theta*data
    tx = tx .- maximum(tx, 1)

    # (Safely) exponentiate and divide by column sums to get p(yⁱ = j | xⁱ,Θ).
    # Each column of p is a categorical distribution (like a histogram whose
    # ncat bins are probabilities summing to 1.)
    etx = exp(tx) # size ncat x m
    p = etx ./ sum(etx, 1)
    # display(p[:,1:10])

    # Cost function including weight decay term
    J = -1/m * sum(sum(ind.*log(p))) + lambda/2*sum(sum(theta.^2))

    # Gradient
    grad = -1/m * (ind - p)*data' + lambda*theta

    # theta = theta[:]
    J, grad[:]
end

function trainSoftmax(theta, ncat, x, y, lambda)
    alg = :LD_LBFGS
    npars = length(theta)
    opt = Opt(alg, npars)
    ftol_abs!(opt, 1e-6)
    ftol_rel!(opt, 1e-6)
    xtol_abs!(opt, 1e-4)
    xtol_rel!(opt, 1e-4)
    maxeval!(opt, 1000)
    lower_bounds!(opt, -5.0*ones(npars))
    upper_bounds!(opt, +5.0*ones(npars))
    println("Using ", algorithm_name(opt))

    # Wrap the cost function to match the signature expected by NLopt
    ncalls = 0
    function f(t::Vector, grad::Vector)
        J, grad[:] = softmaxCost(t,ncat,x,y,lambda)
        
        ncalls += 1
        ng = norm(grad)
        println("$ncalls: J = $J, grad = $ng")
        
        J
    end

    min_objective!(opt, f)
    (minCost, optTheta, status) = optimize!(opt, theta)
    println("Cost = $minCost (returned $status)")
    optTheta, minCost, status
end

function trainAutoencoder(theta,nv,nh,lambda,beta,sparsity,data; maxIter=1000)
    # For algorithm choices, see the NLOPT docs:
    #  http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms
    # For a concise list, view the NLOpt.jl source at 
    #  https://github.com/JuliaOpt/NLopt.jl/blob/master/src/NLopt.jl.
    # e.g. :LD_MMA :LD_SLSQP :LN_SBPLX :LD_TNEWTON :LD_LBFGS
    # The first letter means Global (G) or Local (L).
    # The second letter means D: Derivative(s) required, N: No derivs. required. 
    alg = :LD_LBFGS
    npars = length(theta)
    opt = Opt(alg, npars)
    ftol_abs!(opt, 1e-6)
    ftol_rel!(opt, 1e-6)
    maxeval!(opt, maxIter)
    lower_bounds!(opt, -5.0*ones(npars))
    upper_bounds!(opt, +5.0*ones(npars))
    println("Using ", algorithm_name(opt))

    # Wrap the cost function to match the signature expected by NLopt
    ncalls = 0
    function f(x::Vector, grad::Vector)
        J, grad[:] = saeCost(x,nv,nh,lambda,beta,sparsity,data)
        
        ncalls += 1
        ng = norm(grad)
        println("$ncalls: J = $J, grad = $ng")
        
        J
    end

    min_objective!(opt, f)
    (minCost, optTheta, status) = optimize!(opt, theta)
    println("Cost = $minCost (returned $status)")
    optTheta, minCost, status
end

function viewW1(theta, nh, nv; saveAs = "output/edges.png")
    W1 = reshape(theta[1:nh*nv], nh, nv)
    figure(name="W1 matrix from trained theta")
    display(imagesc(255*W1))
    
    A = tileColumns(W1')
    println("min $(minimum(A)), max $(maximum(A))")
    figure(name="Autoencoder weights")
    display(imagesc(255*A))
    savefig(saveAs)
end





















