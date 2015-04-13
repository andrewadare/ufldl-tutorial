using Images, ImageView
using MNIST


function drawFeatures()
    nv = 28*28                # Nodes in visible layer (pixels in each image)
    nh = 196                  # Nodes in hidden layer of autoencoder
    theta = readdlm("data/mnist_features.txt")
    W1 = reshape(theta[1:nh*nv], nh, nv) # 196x784. Each row is a feature.
    m,n = size(W1)                       # m feature vectors, each length n.
    l = int(sqrt(n))                     # Assume feature array is square (lxl).
    g = int(sqrt(m))                     # Also assume nh is a perfect square.
    ops = Dict(:pixelspacing => [1,1])
    cg = canvasgrid(g, g; w=600, h=600, pad=0, name="Features from digits 5-9")
    k = 1
    for i = 1:g
        for j = 1:g
            ft = reshape(W1[k,:],l,l)'
            img = sc(grayim(ft))
            imgc, imgslice = view(cg[i,j], img; ops...)
            k += 1
        end 
    end
end

function drawDigits()
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

    g = 14 # Draw g x g random digits in a square grid
    ops = Dict(:pixelspacing => [1,1])
    cg1 = canvasgrid(g, g; w=600, h=600, pad=0, name="Digits 5-9 (unlabeled)")
    cg2 = canvasgrid(g, g; w=600, h=600, pad=0, name="Digits 0-4 (test)")
    for i = 1:g
        for j = 1:g

            # Draw unlabeled set 5-9
            k = rand(1:size(xunlabeled,2))
            ft = reshape(xunlabeled[:,k],28,28)'
            img = sc(grayim(ft))
            view(cg1[i,j], img; ops...)

            # Draw test set 0-4
            k = rand(1:size(xtest,2))
            ft = reshape(xtest[:,k],28,28)'
            img = sc(grayim(ft))
            view(cg2[i,j], img; ops...)
        end 
    end
end

drawFeatures()
drawDigits()
