using MAT
using Winston, Color

include ("dltUtils.jl")

function main()

    # 1. Create 10000 8x8 image patches from 10 nature scenes and view a few.
    imFile = matopen("data/IMAGES_RAW.mat") # Provided in tutorial.
    names(imFile)                           # lists one name: "IMAGESr"
    imgs = read(imFile, "IMAGESr")          # 512x512x10 Array{Float64,3}:
    npatches = 10000

    if false
        # Display images with Winston. TODO: set a grayscale colormap
        for i = 1:10
            figure(name="Unprocessed natural image $i")
            display(imagesc(imgs[:,:,i]))
            savefig("output/raw_natural_image_$i.png")
        end
    end

    # Image patch dataset: patchsize^2 x npatches
    x = sampleImages(imgs; patchsize = 12, npatches = npatches, normalize = false)

    # Center x
    mu = mean(x,1)
    for j = 1:npatches
        x[:,j] -= mu[j]
    end

    # PCA
    Σ = x*x' / npatches
    U,S,VT = svd(Σ) # 144x144
    xrot = U'*x

    # Dimensionality reduction: find k that retains 99% of the variance of x.
    k = length(S)
    sumS = sum(S)
    for i = 1:length(S)
        if sum(S[1:i]) >= 0.99*sumS
            k = i
            println("k(99%) = $k")
            break
        end
    end

    xhat = U[:,1:k]*xrot[1:k,:]

    # PCA whitening: standardize the data so the covariance is an identity matrix.
    w = (1 ./ sqrt(S + 1e-5)) .* xrot
    
    # ZCA whitening
    z = U*w


    # Plot ---------------------------------------------------------------------
    nshow = 12
    Winston.colormap(Color.colormap("Grays", 256))
    figure(name="Unprocessed image patches ($nshow x $nshow sample)")
    display(imagesc(255*imageMosaic(x', nshow; random = false)))
    savefig("output/raw_patches.png")

    figure(name="Reduced image patches (k = $k / 144)")
    display(imagesc(255*imageMosaic(xhat', nshow; random = false)))
    savefig("output/k99percent_patches.png")

    figure(name="Covariance of data before PCA (mean subtracted)")
    display(imagesc(x*x'/npatches))
    savefig("output/x_cov.png")

    figure(name="Covariance of data after PCA")
    display(imagesc(xrot*xrot'/npatches))
    savefig("output/xrot_cov.png")

    figure(name="Covariance of PCA Whitened Data with regularization")
    display(imagesc(w*w'/npatches))
    savefig("output/pca_covariance.png")
    
    figure(name="Covariance of ZCA Whitened Data")
    display(imagesc(z*z'/npatches))
    savefig("output/zca_covariance.png")

    figure(name="ZCA image patches")
    display(imagesc(255*imageMosaic(z', nshow; random = false)))
    savefig("output/zca_whitened_patches.png")

end

main()