## Utility/common functions for deep learning tutorial
## Material from http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial
## and Dan Luu's solutions at https://github.com/danluu/UFLDL-tutorial.git
## Implemented in julia by Andrew Adare as an educational exercise.

# Logistic sigmoid function and its gradient
sigmoid(z) = 1 ./ (1 + exp(-z))
sigmoidGradient(z) = sigmoid(z).*(1 - sigmoid(z))

# Kullback-Leibler divergence between Bernoulli variables with means p and q.
klBernoulli(p,q) = sum(p.*log(p./q) + (1-p).*log((1-p)./(1-q)))

function sampleImages(; patchsize = 8, npatches = 10000)
    imFile = matopen("data/IMAGES.mat") # 10 nature scenes provided in tutorial.
    # names(imFile)                     # lists one name: "IMAGES"
    imgs = read(imFile, "IMAGES")       # 512x512x10 Array{Float64,3}:

    if false
        # Display images with Winston. Images were PCA'ed. 
        # With this colormap + preprocessing, they don't look very "natural".
        for i = 1:10
            figure(name="Natural image $i")
            display(imagesc(imgs[:,:,i]))
            savefig("output/natural_image_$i.png")
        end
    end

    # Sample random patches from larger images
    patches = zeros(patchsize*patchsize, npatches)
    r,c,_ = size(imgs)
    for i = 1:npatches
        # Pick a random upper left corner and get the patch
        x = rand(1:c-patchsize+1) # 512-8+1
        y = rand(1:r-patchsize+1) # 512-8+1
        patch = imgs[x:x+7,y:y+7,rand(1:10)]
        patches[:,i] = patch[:]
    end

    # For the autoencoder to work well we need to normalize the data
    # Specifically, since the output of the network is bounded between [0,1]
    # (due to the sigmoid activation function), we have to make sure 
    # the range of pixel values is also bounded between [0,1].

    # Center
    patches -= mean(patches)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    d = 3*std(patches)
    patches = max(min(patches, d), -d) / d

    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches + 1) * 0.4 + 0.1
end

# Return a square mosaic of N² images selected randomly from an m×n dataset X
# where m is the number of images and each image is √n x √n.
function imageMosaic(X, N)
    m,n = size(X)
    h,w = int(sqrt(n)), int(sqrt(n))
    mosaic = zeros(Float64, N*h, N*w)
    for i in 1:N
        for j in 1:N
            digit = reshape(X[rand(1:m),:], h, w)'
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

    # Backpropagation ----------------------------------------------------------
    # Sparsity penalty to be added to delta2. 
    # This is a vector; repeat m times when computing delta2.
    kld = (1-rho)./(1-rhoHat) - rho ./ rhoHat

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
