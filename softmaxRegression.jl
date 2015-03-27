using MNIST
using Winston
using Color

include("dltUtils.jl")

function main()
    ncat = 10                 # Number of classification categories
    nin = 28*28               # Input size (number of pixels in each image)
    lambda = 1e-4             # Regularization parameter
    checkGradient = false

    # Get handwritten digit images and their labels from MNIST package.
    x, y = traindata()        # 784x60000, 60000 
    xtest, ytest = testdata() # 784x10000, 10000

    # The pixel intensities are provided in [0,255]. Rescale to [0,1].
    x /= 255.
    xtest /= 255.

    # Remap 0 -> 10 in the labels for one-hot encoding (since 1-based indexing)
    y[findin(y, [0.0])] = 10.0
    ytest[findin(ytest, [0.0])] = 10.0

    # Initialize theta with random Gaussian weights, sigma 0.005.
    theta = 0.005*randn(ncat*nin)
    J, grad = softmaxCost(theta, ncat, x, y, lambda)

    if checkGradient   
        nin = 8;
        x = randn(nin, 100)
        y = rand(1:10, 100)

        theta = 0.005*randn(ncat*nin)
        J, grad = softmaxCost(theta, ncat, x, y, lambda)

        numgrad = numericalGradient(t->softmaxCost(t,ncat,x,y,lambda), theta)
        display([numgrad grad])
        diff = norm(numgrad-grad)/norm(numgrad+grad)
        report = diff < 1e-9 ? "PASS" : "FAIL"
        println("$report: Numerical - analytic gradient norm difference = $diff.")
        return
    end

    # Train
    trainedTheta, minCost, status = trainSoftmax(nin, ncat, x, y, lambda)

    # Test
    trainedTheta = reshape(trainedTheta, ncat, nin)
    probs = trainedTheta*xtest
    preds = [indmax(probs[:,j]) for j = 1:size(probs,2)]
    accuracy = 100*mean(preds .== ytest)
    println("Accuracy: $accuracy%")

    # Plot ---------------------------------------------------------------------
    Winston.colormap(Color.colormap("Grays", 256))
    digit = reshape(x[:,2], 28, 28)
    figure(name="A random image")
    display(imagesc(255*digit))
    # savefig("output/raw_patches.png")
end

main()
