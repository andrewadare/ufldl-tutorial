using MNIST
# using Winston

include("dltUtils.jl")

function main()
    ncat = 10                 # Number of categories (digits 0-9)
    nin = 28*28               # Nodes in input layer (pixels in each image)
    nh1 = 196                 # Nodes in hidden layer of autoencoder 1
    nh2 = 196                 # Nodes in hidden layer of autoencoder 2
    lambda = 3e-3             # Regularization parameter
    sparsity = 0.1            # Avg activation of hidden units
    beta = 3                  # Weight of sparsity penalty term
    aeTrainIter = 2          # Number of L-BFGS iterations for AE training

    # Get handwritten digit images and their labels from MNIST package.
    x, y = traindata()        # 784x60000, 60000 
    xtest, ytest = testdata() # 784x10000, 10000

    # The pixel intensities are provided in [0,255]. Rescale to [0,1].
    x /= 255.
    xtest /= 255.

    # Remap 0 -> 10 in the labels for one-hot encoding (since 1-based indexing)
    y[findin(y, [0.0])] = 10.0
    ytest[findin(ytest, [0.0])] = 10.0

    tic()
    println("Training sparse autoencoder 1...")
    sae1Theta = initWeights(nin, nh1)
    sae1OptTheta, optJ1, status1 = 
    trainAutoencoder(sae1Theta,nin,nh1,lambda,beta,sparsity,x; 
                     maxIter=aeTrainIter)
    toc()

    println("Feed-forward pass to compute features from autoencoder 1...")
    sae1Features = feedForwardAutoencoder(sae1OptTheta, nin, nh1, x)

    tic()
    println("Training sparse autoencoder 2...")
    sae2Theta = initWeights(nh1, nh2)
    sae2OptTheta, optJ2, status2 = 
    trainAutoencoder(sae2Theta,nh1,nh2,lambda,beta,sparsity,sae1Features; 
                     maxIter=aeTrainIter)
    toc()

    println("Feed-forward pass to compute features from autoencoder 2...")
    sae2Features = feedForwardAutoencoder(sae2OptTheta, nh1, nh2, sae1Features)

    println("Training softmax classifier on stacked autoencoder output...")
    saeSoftmaxOptTheta, _, _ = trainSoftmax(nh2, ncat, sae2Features, y, lambda)

    # Up to this point, we have called functions from previous exercises. Now 
    # for some new code.

    stack = Vector{AEWeights}(2)
    stack[1] = AEWeights(reshape(sae1OptTheta[1:nin*nh1], nh1, nin), 
                         sae1OptTheta[2*nin*nh1+1:2*nin*nh1+nh1])
    stack[2] = AEWeights(reshape(sae2OptTheta[1:nh1*nh2], nh2, nh1),
                         sae2OptTheta[2*nh1*nh2+1:2*nh1*nh2+nh2])

    stackpars, arch = stack2Pars(stack)
    println("$(arch.inputSize) input units. Hidden layer sizes:")
    display(arch.layerSizes)

    stackedAeTheta = [saeSoftmaxOptTheta; stackpars]

    size(stackedAeTheta)
    trainStackedAutoencoder(stackedAeTheta, nin, nh2, ncat, arch, lambda,
                            x, y; aeTrainIter)
end

main()
