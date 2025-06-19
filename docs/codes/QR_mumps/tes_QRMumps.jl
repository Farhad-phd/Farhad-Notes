using QRMumps, LinearAlgebra, SparseArrays, Printf, ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools, LinearAlgebra

# this package
using Revise
using JSOSolvers

# setting the problem 
F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
model = ADNLSModel(F, [-1.2; 1.0], 2)
meta_nls = nls_meta(model)
# original sizes
m = meta_nls.nequ    # number of residual‐equations
n = meta_nls.nvar    # number of variables


σ = 5.0
lower_end = sqrt(σ) * I(n)
lower_end_b = zeros(n)



nnzj = meta_nls.nnzj # number of nonzeros in the jacobian
irn = Vector{Int}(undef, nnzj)
jcn = Vector{Int}(undef, nnzj)
jac_structure_residual!(model, irn, jcn);
println("Jacobian structure:")
println("irn: ", irn)
println("jcn: ", jcn)



x = get_x0(model)
vals = similar(x, nnzj)
jac_coord_residual!(model, x, vals)
println("Jacobian values:")
println(vals)

# println("using jac_op_residual")
# Jv = similar(model.meta.x0, meta_nls.nequ)
# Jtv = similar(model.meta.x0, meta_nls.nvar)
# Jx = jac_op_residual!(model, irn, jcn, vals, Jv, Jtv)

# mat_Jx = Matrix(Jx)
# println("Jacobian matrix Jx:")
# println(mat_Jx)




# 1) compute the new row‐indices: m+1, m+2, …, m+n
new_rows = m .+ (1:n)

# 2) the new col‐indices are just 1:n (identity)
new_cols = 1:n

# 3) the diagonal‐values are all sqrt(σ)
new_vals = fill(sqrt(σ), n)

# 4) append them
append!(irn, new_rows)
append!(jcn, new_cols)
append!(vals, new_vals)

# now (irn, jcn, vals) represents the (m+n)×n matrix [ J(x) ; √σ·I ]
# you can build the sparse A_aug if you like:
A_aug = sparse(irn, jcn, vals, m+n, n)

println("Augmented row‐indices:", irn)
println("Augmented col‐indices:", jcn)
println("Augmented values:     ", vals)
println("Augmented sparse A_aug:")
display(A_aug)


# 1) build A_aug
A_aug = sparse(irn, jcn, vals, m+n, n)

# 2) build b_aug
r      = F(get_x0(model))           # m-vector residual
lower_b = zeros(n)                  # n-vector
b_aug  = vcat(r, lower_b)           # (m+n)-vector

# 3) solve with QRMumps in one go
qrm_init()
spmat   = qrm_spmat_init(A_aug)
x_delta = qrm_least_squares(spmat, b_aug)

println("Solution x =", x_delta)


#version 2 

# 1) initialize the library
qrm_init()

# 2) create the QRMumps sparse‐matrix object from your triplets
spmat = qrm_spmat_init(m + n, n, irn, jcn, val; sym=false)

# 3) (optional) if you’re going to do many right‐hand sides:
spfct = qrm_spfct_init(spmat)
qrm_analyse!(spmat, spfct)
qrm_factorize!(spmat, spfct)

# 4) form your RHS:
r      = F(get_x0(model))                   # m-vector residual
lower_b = zeros(n)                          # the "zero" part for √σ·I
b_aug  = vcat(r, lower_b)                   # (m+n)-vector

# 5a) one‐shot solve 
x_sol = qrm_least_squares(spmat, b_aug)

# 5b) or, if you did steps 3):
z     = qrm_apply(spfct, b_aug, transp='n') # compute Qᵀ b
x_sol = qrm_solve(spfct, z,   transp='n')   # solve R x = z

println("Least‐squares solution δx = ", x_sol)




#################################

using QRMumps, LinearAlgebra, SparseArrays
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools
using Revise, JSOSolvers

# ----------------------------------------
# 1) Problem setup
# ----------------------------------------
σ = 5.0

# Define residual F : ℝ² → ℝ²
F(x) = @SVector [ x[1] - 1;
                  2*(x[2] - x[1]^2) ]

# Create the ADNLS model, get metadata
model    = ADNLSModel(F, [-1.2; 1.0], 2)
meta_nls = nls_meta(model)
m, n     = meta_nls.nequ, meta_nls.nvar

# ----------------------------------------
# 2) Jacobian structure & values
# ----------------------------------------
# allocate arrays for COO pattern
nnzj = meta_nls.nnzj
rows = Vector{Int}(undef, nnzj)
cols = Vector{Int}(undef, nnzj)

# fill in the sparsity pattern of J(x)
jac_structure_residual!(model, rows, cols)

# evaluate at initial x0 to get the values
x0   = get_x0(model)
vals = similar(rows, Float64)
jac_coord_residual!(model, x0, vals)

# copy into working triplet arrays
irn = copy(rows)
jcn = copy(cols)
val = copy(vals)

# ----------------------------------------
# 3) Append the √σ·Iₙ block in COO form
# ----------------------------------------
# identity block lives in rows m+1 .. m+n
new_rows = m .+ (1:n)                 # vector of length n
new_cols = collect(1:n)               # same columns 1..n
new_vals = fill(sqrt(σ), n)           # diagonal entries

append!(irn, new_rows)
append!(jcn, new_cols)
append!(val, new_vals)

# ----------------------------------------
# 4) Build RHS and solve with QRMumps
# ----------------------------------------
# original residual vector
r      = F(x0)                        # length-m
lower_b = zeros(n)                    # length-n
b_aug  = vcat(r, lower_b)             # length (m+n)

# initialize the QRMumps library
qrm_init()

# build the QRMumps sparse‐matrix object from triplets
#   (m+n) rows, n columns, unsymmetric → sym=false
spmat = qrm_spmat_init(m+n, n, irn, jcn, val; sym=false)

# (optional) do analysis + factorization separately:
spfct = qrm_spfct_init(spmat)
qrm_analyse!(spmat, spfct)
qrm_factorize!(spmat, spfct)

# solve: form Qᵀ b and then R x = Qᵀ b
z     = qrm_apply(spfct, b_aug, transp='n')
x_sol = qrm_solve(spfct, z,   transp='n')

println("Least‐squares solution δx = ", x_sol)

# ----------------------------------------
# 5) Cleanup
# ----------------------------------------
qrm_spfct_free(spfct)
qrm_spmat_free(spmat)
qrm_finalize()

####################################################################


