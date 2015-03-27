using MNIST
using Winston

include("dltUtils.jl")

function main()
    ncat = 5                  # Number of categories (digits 0-4)
    nv = 28*28                # Nodes in visible layer (pixels in each image)
    nh = 196                  # Nodes in hidden layer of autoencoder
    lambda = 3e-3             # Regularization parameter
    sparsity = 0.1            # Avg activation of hidden units
    beta = 3                  # Weight of sparsity penalty term

    # Get handwritten digit images and their labels from MNIST package.
    x, y = traindata()        # 784x60000, 60000 

    # The pixel intensities are provided in [0,255]. Rescale to [0,1].
    x /= 255.

    # Self-taught learning: train the autoencoder on features learned from
    # the subset of digits 5-9, which we pretend are unlabeled. Select the
    # set of indices in the labeled and "unlabeled" categories.
    labeled, nolabel = find(y.<=4), find(y.>=5)
    
    # Split the labeled array of indices in two - half for training, 
    # half for testing
    ntrain  = round(length(labeled)/2)
    train, test = labeled[1:ntrain], labeled[ntrain+1:end]

    # Subdivide the data and labels
    xtrain = x[:,train]
    xtest  = x[:,test]
    ytrain = y[train] + 1 # Shift up for 1-hot encoding (1-based indexing)
    ytest  = y[test] + 1
    xunlabeled = x[:,nolabel]

    println("$(length(nolabel)) examples in unlabeled set.")
    println("$(length(train)) examples in supervised training set.")
    println("$(length(test)) examples in supervised testing set.")

    theta = initWeights(nv, nh)
    tic()
    optTheta, optJ, aestatus = 
    trainAutoencoder(theta,nv,nh,lambda,beta,sparsity,xunlabeled; maxIter=400)
    toc()

    Winston.colormap(Color.colormap("Grays", 256))
    viewW1(optTheta, nh, nv; saveAs="output/mnist_features.png")

    trainFeatures = feedForwardAutoencoder(optTheta, nv, nh, xtrain)
    testFeatures  = feedForwardAutoencoder(optTheta, nv, nh, xtest)

    # Train softmax classifier
    trainedTheta, _, _ = trainSoftmax(nh, ncat, trainFeatures, ytrain, 1e-4)

    # Make predictions on features learned from test data
    trainedTheta = reshape(trainedTheta, ncat, nh)
    probs = trainedTheta*testFeatures
    preds = [indmax(probs[:,j]) for j = 1:size(probs,2)]
    accuracy = 100*mean(preds .== ytest)
    println("Accuracy: $accuracy%")

end

main()
