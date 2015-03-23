using MAT
using Winston

include("dltUtils.jl")

function main()
    nv = 8*8                # Units in the visible layers. Equal to patch size.
    nh = 25                 # Units in the hidden layer.
    sparsity = 0.01         # Target mean activation of hidden units.
    lambda = 1e-4           # Weight decay (regularization) parameter.
    beta = 3                # Sparsity penalty weight.

    # 1. Create 10000 8x8 image patches from 10 nature scenes and view a few.
    patches = sampleImages(patchsize = 8, npatches = 10000)
    if true
        mosaic = imageMosaic(patches', 10)
        figure(name="Image patches (10x10 sample)")
        display(imagesc(mosaic))
        savefig("output/patches.png")
    end

    # 2. Sparse autoencoder objective function and its gradient wrt theta.
    # The weights W1,W2,b1,and b2 are unrolled into theta as a long vector.
    theta = initWeights(nv, nh)
    J, grad = saeCost(theta, nv, nh, lambda, beta, sparsity, patches)

    # 3. Gradient checking
    # Compare "analytic" gradient with numerical approximation.
    # Before running the AE test, reduce npatches above to 1000 or so.
    println("Analytic vs numeric gradients for a small test problem:")
    checkGradient()
    if false
        println("Analytic vs numeric gradients for autoencoder function:")
        cost(t) = saeCost(t,nv,nh,lambda,beta,sparsity,patches)
        numgrad = numericalGradient(cost, theta)
        display([numgrad grad])
        diff = norm(numgrad-grad)/norm(numgrad+grad)
        report = diff < 1e-9 ? "PASS" : "FAIL"
        println("$report: Numerical - analytic gradient norm difference = $diff.")
    end
end

main()
