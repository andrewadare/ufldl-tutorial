# Principal Components Analysis (simple 2D exercise)

using Winston

plot = true
save = false

function main()
x = readdlm("data/pcaData.txt") # 2x45
m = size(x,2)

# Center x at (0,0).
mu = mean(x,2)
println("Mean of dataset: $mu")
for j = 1:m
    x[:,j] -= mu
end

# Find eigenvalues/vectors of the covariance matrix.
# The principal axes are the columns of U.
Σ = x * x' / m
U,S,VT = svd(Σ) # 2x2, 2x1, 2x2. (Note that S is returned as a vector here).

# Rotate x to lie along the principal axes u1 and u2.
# This decorrelates the data: its covariance matrix is now diagonal (in fact S).
xrot = U'*x
println("λ1, λ2 = $S. Covariance matrix of rotated data:")
display(xrot*xrot'/m)

# Dimensionality reduction: truncate to U to the first k columns (size 1x45)
k = 1
xtr = xrot[1:k,:]

# Rotate back to original x1,x2 basis (size 2x45)
xt = U[:,1:k]*xtr

# PCA whitening: standardize the data so the covariance is an identity matrix.
w = (1 ./ sqrt(S + 1e-5)) .* xrot
println("\nCovariance matrix of PCA whitened data:")
display(w*w'/m)

# ZCA whitening:
z = U*w
println("\nCovariance matrix of ZCA whitened data:")
display(z*z'/m)

# Plotting ---------------------------------------------------------------------
plot || return

# Draw unmodified data
figure(name="Original Data")
fp1 = FramedPlot(xlabel="x_1", ylabel="x_2")
add(fp1, Points(x[1,:], x[2,:], color="DodgerBlue", kind="circle"))
display(fp1)
save && savefig("output/original_data.pdf")

# Superimpose principal axis vectors. The scale of the eigenvectors is not
# meaningful, so I scale u1 by (-1) for a more "conventional" appearance.
figure(name="Principal Axes")
add(fp1, 
    Curve([0 -1*U[1,1]], [0 -1*U[2,1]]), PlotLabel(0.8, 0.7, "u_1"),
    Curve([0    U[1,2]], [0    U[2,2]]), PlotLabel(0.2, 0.7, "u_2"))
display(fp1)
save && savefig("output/principal_axes.pdf")

# Plot xrot
figure(name="Rotated Data")
fp2 = FramedPlot(xlabel="(U^Tx)_1", ylabel="(U^Tx)_2")
add(fp2, Points(xrot[1,:], xrot[2,:], color="FireBrick", kind="circle"))
display(fp2)
save && savefig("output/rotated_data.pdf")

# Reduced data
figure(name="Reduced Rotated Data")
display(scatter(xtr, zeros(xtr), 
        color="DarkSlateBlue",
        xlabel="(U^Tx)_1", 
        ylabel="(U^Tx)_2"))
save && savefig("output/reduced_rotated_data.pdf")

# Reduced data, rotated back to original frame
figure(name="Reduced Data")
display(scatter(xt[1,:], xt[2,:], xlabel="x_1", ylabel="x_2"))
save && savefig("output/reduced_data.pdf")

figure(name="PCA Whitened Data")
display(scatter(w[1,:], w[2,:], xlabel="w_1", ylabel="w_2"))
save && savefig("output/pca_whitened_data.pdf")

figure(name="ZCA Whitened Data")
display(scatter(z[1,:], z[2,:], xlabel="z_1", ylabel="z_2"))
save && savefig("output/zca_whitened_data.pdf")
end

main()