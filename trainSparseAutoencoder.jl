using MAT
using Winston, Color
using NLopt

include("dltUtils.jl")

function main()
    nv = 8*8                # Units in the visible layers. Equal to patch size.
    nh = 25                 # Units in the hidden layer.
    sparsity = 0.01         # Target mean activation of hidden units.
    lambda = 1e-4           # Weight decay (regularization) parameter.
    beta = 3                # Sparsity penalty weight.
    npatches = 10000
    Winston.colormap(Color.colormap("Grays", 256))

    # 1. Create 10000 8x8 image patches from 10 nature scenes and view a few.
    imFile = matopen("data/IMAGES.mat") # 10 nature scenes provided in tutorial.
    names(imFile)                       # lists one name: "IMAGES"
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

    patches = sampleImages(imgs; patchsize = 8, npatches = npatches)
    if true
        nshow = 15
        mosaic = imageMosaic(patches', nshow)
        figure(name="Image patches ($nshow x $nshow sample)")
        display(imagesc(255*mosaic))
        savefig("output/patches.png")
    end

    # 2. Sparse autoencoder objective function and its gradient wrt theta.
    # The weights W1,W2,b1,and b2 are unrolled into theta as a long vector.
    theta = initWeights(nv, nh)
    println("Initialized random weights: ", mean(theta), " ", std(theta))
    J, grad = saeCost(theta, nv, nh, lambda, beta, sparsity, patches)

    println("J = $J")

    # 3. Gradient checking
    # Compare "analytic" gradient with numerical approximation.
    # Reduce npatches for this check, otherwise it is too slow.
    if npatches <= 1000
        println("Analytic vs numeric gradients for a small test problem:")
        checkGradient()
    
        println("Analytic vs numeric gradients for autoencoder function:")
        cost(t) = saeCost(t,nv,nh,lambda,beta,sparsity,patches)
        numgrad = numericalGradient(cost, theta)
        display([numgrad grad])
        diff = norm(numgrad-grad)/norm(numgrad+grad)
        report = diff < 1e-9 ? "PASS" : "FAIL"
        println("$report: Numerical - analytic gradient norm difference = $diff.")
    end

    # 4. Train sparse autoencoder
    optTheta, optJ, status = 
    trainAutoencoder(theta,nv,nh,lambda,beta,sparsity,patches; maxIter=400)
    
    viewW1(optTheta, nh, nv; saveAs="output/edges.png")
end


main()
