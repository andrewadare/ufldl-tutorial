using MAT
using Winston
using NLopt

include("dltUtils.jl")

function main()
    nv = 8*8                # Units in the visible layers. Equal to patch size.
    nh = 25                 # Units in the hidden layer.
    sparsity = 0.01         # Target mean activation of hidden units.
    lambda = 1e-4           # Weight decay (regularization) parameter.
    beta = 3                # Sparsity penalty weight.
    npatches = 10000

    # 1. Create 10000 8x8 image patches from 10 nature scenes and view a few.
    patches = sampleImages(patchsize = 8, npatches = npatches)
    if true
        nshow = 15
        mosaic = imageMosaic(patches', nshow)
        figure(name="Image patches ($nshow x $nshow sample)")
        display(imagesc(mosaic))
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
    xtol_abs!(opt, 1e-4)
    xtol_rel!(opt, 1e-4)
    maxeval!(opt, 1000)
    lower_bounds!(opt, -5.0*ones(npars))
    upper_bounds!(opt, +5.0*ones(npars))
    println("Using ", algorithm_name(opt))

    # Wrap the cost function to match the signature expected by NLopt
    ncalls = 0
    function f(x::Vector, grad::Vector)
        J, grad[:] = saeCost(x,nv,nh,lambda,beta,sparsity,patches)
        
        ncalls += 1
        ng = norm(grad)
        println("$ncalls: J = $J, grad = $ng")
        
        J
    end

    min_objective!(opt, f)
    (minCost, optTheta, status) = optimize!(opt, theta)
    println("Cost = $minCost (returned $status)")
    
    viewW1(optTheta, nh, nv)
end

function viewW1(theta, nh, nv)
    W1 = reshape(theta[1:nh*nv], nh, nv)
    figure(name="W1 matrix from trained theta")
    display(imagesc(W1))
    
    A = tileColumns(W1')
    figure(name="Autoencoder weights")
    display(imagesc(A))
    savefig("output/edges.png")
end


main()
