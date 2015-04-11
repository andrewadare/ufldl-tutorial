using MAT
using Images, ImageView

include("dltUtils.jl")

const nChannels = 3                # r,g,b
const imgDim = 64                  # Images are 64x64
const patchDim = 8                 # Patches are 8x8
const poolDim = 19                 # Size of pooling region
const nPatches = 50_000            
const nv = patchDim^2 * nChannels
const nh = 400
const epsilon = 0.1

function main()
    optTheta  = readdlm("data/stl10features.txt")
    meanPatch = readdlm("data/patch_means.txt")
    zcaWhite  = readdlm("data/zcawhite.txt")

    W = reshape(optTheta[1:nv * nh], nh, nv)
    b = optTheta[2*nh*nv+1:2*nh*nv+nh]
    displayColorNetwork((W*zcaWhite)')

    imFile = matopen("data/stlTrainSubset.mat") # List using names(imFile)
    trainLabels = read(imFile, "trainLabels")
    trainImages = read(imFile, "trainImages")
    nTrain = read(imFile, "numTrainImages")     # 2000

    # Test convolution code on the first 8 images 
    # (convImages is 64 x 64 x 3 x 8).
    convImages = trainImages[:, :, :, 1:8]
    # for i = 1:8
    #     view(convImages[:,:,:,i])
    # end

    convolvedFeatures = cnnConvolve(patchDim, nh, convImages, W, b, zcaWhite, meanPatch)

    # Test the cnnConvolve function
    for i = 1:1 # Change to 1:1000 to run test
        fNum  = rand([1:nh;])
        imNum = rand([1:8;])
        imRow = rand([1:imgDim - patchDim + 1;])
        imCol = rand([1:imgDim - patchDim + 1;])    
     
        patch = convImages[imRow:imRow + patchDim - 1, 
                           imCol:imCol + patchDim - 1, :, imNum]
        patch = patch[:]            
        patch = patch - meanPatch
        patch = zcaWhite * patch
      
        features = feedForwardAutoencoder(optTheta, nv, nh, patch)
        diff = features[fNum, 1] - convolvedFeatures[fNum, imNum, imRow, imCol]
        println("Convolved feature - autoencoder activation: $diff")
        if abs(diff) > 1e-9
            println("Convolved feature does not match activation from autoencoder")
            println("Feature Number    $fNum")
            println("Image Number      $imNum")
            println("Image Row         $imRow")
            println("Image Column      $imCol")
            println("Convolved feature $(convolvedFeatures(fNum, imNum, imRow, imCol)))")
            println("Sparse AE feature $(features(fNum, 1)))")
            error("Convolved feature does not match activation from autoencoder")
        end 
    end

    pooledFeatures = cnnPool(poolDim, convolvedFeatures)

    # Test pooling function
    testMatrix = reshape(1:64., 8, 8)
    expectedMatrix = [mean(mean(testMatrix[1:4, 1:4])) 
                      mean(mean(testMatrix[5:8, 1:4]))
                      mean(mean(testMatrix[1:4, 5:8]))
                      mean(mean(testMatrix[5:8, 5:8]))]
    testMatrix = reshape(testMatrix, 1, 1, 8, 8)
    pooledFeatures = cnnPool(4, testMatrix)[:]
    if isequal(pooledFeatures, expectedMatrix)
        println("Congratulations! Your pooling code passed the test.")
    else
        println("Pooling incorrect")
        println("Expected")
        println(expectedMatrix)
        println("Got")
        println(pooledFeatures)
    end

    # TODO: run convolution and pooling on data.
end

main()