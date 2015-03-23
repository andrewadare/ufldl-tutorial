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
    ftol_abs!(opt, 1e-5)
    ftol_rel!(opt, 1e-4)
    maxeval!(opt, 300)
    lower_bounds!(opt, -10.0*ones(npars))
    upper_bounds!(opt, +10.0*ones(npars))
    println("Using ", algorithm_name(opt))

    # Create a wrapper for the cost function acceptable to NLopt
    function f(x::Vector, grad::Vector)
        J,grad = saeCost(x,nv,nh,lambda,beta,sparsity,patches)
        J
    end

    min_objective!(opt, f)

    # min_objective!(opt, (theta, grad) -> lrCost!(theta, X, t, lambda, grad))
    (minCost, optTheta, status) = optimize!(opt, zeros(npars))        
    println("Cost = $minCost (returned $status)")


end

main()
