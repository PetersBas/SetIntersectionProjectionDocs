# Tutorial example

[Julia script for this example](https://github.com/slimgroup/SetIntersectionProjection.jl/blob/master/examples/projection_intersection_2D.jl)

We show how we can compute projections onto an intersection of multiple sets using PARSDMM, parallel PARSDMM, multilevel PARSDMM and multilevel parallel PARSDMM.


```julia
@everywhere using SetIntersectionProjection
using PyPlot #optional, for plotting only

mutable struct compgrid #define a computational grid information type
  d :: Tuple
  n :: Tuple
end
```

```julia
#PARSDMM options:
options    = PARSDMM_options() #fill all options with defaults
options.FL = Float32 

#other options include
  x_min_solver          :: String       = "CG_normal" #what algorithm to use for the x-minimization (CG applied to normal equations)
  maxit                 :: Integer      = 200         #max number of PARSDMM iterations
  evol_rel_tol          :: Real         = 1e-3        #stop PARSDMM if ||x^k - X^{k-1}||_2 / || x^k || < options.evol_rel_tol AND options.feas_tol is reached
  feas_tol              :: Real         = 5e-2        #stop PARSDMM if the transform-domain relative feasibility error is < options.feas_tol AND options.evol_rel_tol is reached
  obj_tol               :: Real         = 1e-3        #optional stopping criterion for change in distance from point that we want to project
  rho_ini               :: Vector{Real} = [10.0]      #initial values for the augmented-Lagrangian penalty parameters. One value in array or one value per constraint set in array
  rho_update_frequency  :: Integer      = 2           #update augmented-Lagrangian penalty parameters and relaxation parameters every X number of PARSDMM iterations
  gamma_ini             :: Real         = 1.0         #initial value for all relaxation parameters (scalar)
  adjust_rho            :: Bool         = true        #adapt augmented-Lagrangian penalty parameters or not
  adjust_gamma          :: Bool         = true        #adapt relaxation parameters in PARSDMM
  adjust_feasibility_rho:: Bool         = true        #adapt augmented-Lagrangian penalty parameters based on constraint set feasibility errors (can be used in combination with options.adjust_rho)
  Blas_active           :: Bool         = true        #use direct BLAS calls, otherwise the code will use Julia loop-fusion where possible
  feasibility_only      :: Bool         = false       #drop distance term and solve a feasibility problem
  FL                    :: DataType     = Float32     #type of Float: Float32 or Float64
  parallel              :: Bool         = false       #compute proximal mappings, multiplier updates, rho and gamma updates in parallel
  zero_ini_guess        :: Bool         = true        #zero initial guess for primal, auxiliary, and multipliers
```


```
#select working precision
if options.FL==Float64
  TF = Float64
  TI = Int64
elseif options.FL==Float32
  TF = Float32
  TI = Int32
end
```

Set other performance related items that are not `SetIntersectionProjection` options. `SetIntersectionProjection` solution times will be affected by these. Do not forget to set `export JULIA_NUM_THREADS=2` or any other number of Julia threads that you want to use. In Julia you can check how many Julia threads will be used using the command `Threads.nthreads()`  .

```julia
set_zero_subnormals(true)
BLAS.set_num_threads(2)
FFTW.set_num_threads(2)
```

Load an image or model that we want to projection onto the intersection.

```julia
m = randn(300,400) #load image to project

#set up computational grid (25 and 6 m are the original distances between grid points)
comp_grid = compgrid((TF(25.0), TF(6.0)),(size(m,1), size(m,2))) #set up as compgrid( (dx,dy,dz) , (nx,ny,nz) )
#use as comp_grid.n and comp_grid.d

m = convert(Vector{TF},vec(m)) #PARSDMM needs the 2D/2D model as a vector, also in the right precision

```

We are ready to select the constraints that we want to use. The following block of code sets up and generates the linear operators and projectors onto each set separately. This part is optional, you can also use your own scripts to generate pairs of linear operators and projectors.

```julia
#constraints
constraint = Vector{SetIntersectionProjection.set_definitions}() #Initialize constraint information

#bounds:
m_min     = 1500.0
m_max     = 4500.0
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


#slope constraints (vertical)
m_min     = 0.0 #transform-domain lower bound (may be a vector for bound constraints)
m_max     = 1e6 #transform-domain upper bound (may be a vector for bound constraints)
set_type  = "bounds"
TD_OP     = "D_z" #choose any of the operators listed on the main page ("D_x", "D_z", "D2D", "identity", “DCT”, "DFT", "curvelet")
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


options.parallel       = false #indicate we are working in serial (1 Julia worker, but still use multi-threading)


(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL) 			 #get projectors onto simple sets, linear operators, set information
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options) #precompute and distribute quantities once, reuse later
```

Some properties of each set are stored in `set_Prop`. For example, we can see if set number 2 is non-convex using `set_Prop.ncvx[2]`. To see what set number 2 is, as generated by our scripts above, we can use `set_Prop.tag[2]`, which returns a tuple of (set description, linear operator description).

```julia
           ncvx       ::Vector{Bool}                                          #is the set non-convex? (true/false)
           AtA_diag   ::Vector{Bool}                                          #is A^T A a diagonal matrix?
           dense      ::Vector{Bool}                                          #is A a dense matrix?
           TD_n       ::Vector{Tuple}                                         #the grid dimensions in the transform-domain.
           tag        ::Vector{Tuple{String,String}}                          #(constraint type, linear operator description)
           banded     ::Vector{Bool}                                          #is A a banded matrix?
           AtA_offsets::Union{Vector{Vector{Int32}},Vector{Vector{Int64}}}    #only required if A is banded. A vector of indices of the non-zero diagonals, where the main diagonal is index 0
```

We are ready to solve the projection problem, using the projectors and linear operators generated as above, or using your own.
```julia
#project onto intersection, twice to get accurate timing
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
```

Visualize the results.

```julia
#define axis limits and colorbar limits
xmax = comp_grid.d[1]*comp_grid.n[1]
zmax = comp_grid.d[2]*comp_grid.n[2]
vmi  = 1500
vma  = 4500

#project original model and the one that is projected onto the intersection
figure();imshow(reshape(m,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
savefig("original_model.png",bbox_inches="tight")
figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")
savefig("projected_model.png",bbox_inches="tight")


#plot PARSDMM logs
figure();
subplot(3, 3, 3);semilogy(log_PARSDMM.r_pri)          ;title("r primal")
subplot(3, 3, 4);semilogy(log_PARSDMM.r_dual)         ;title("r dual")
subplot(3, 3, 1);semilogy(log_PARSDMM.obj)            ;title(L"$ \frac{1}{2} || \mathbf{m}-\mathbf{x} ||_2^2 $")
subplot(3, 3, 2);semilogy(log_PARSDMM.set_feasibility);title("TD feasibility violation")
subplot(3, 3, 5);plot(log_PARSDMM.cg_it)              ;title("nr. of CG iterations")
subplot(3, 3, 6);semilogy(log_PARSDMM.cg_relres)      ;title("CG rel. res.")
subplot(3, 3, 7);semilogy(log_PARSDMM.rho)            ;title("rho")
subplot(3, 3, 8);plot(log_PARSDMM.gamma)              ;title("gamma")
subplot(3, 3, 9);semilogy(log_PARSDMM.evol_x)         ;title("x evolution")
tight_layout()
#tight_layout(pad=0.0, w_pad=0.0, h_pad=1.0)
savefig("PARSDMM_logs.png",bbox_inches="tight")
```

Now we project onto the same intersection, but we solve the sub-problems in parallel. This means we compute the ``y_i``-minimization (proximal maps), ``v_i``-update (Lagrangian multipliers), ``\rho_i`` (augmented-Lagrangian penalty parameters), and ``\gamma_i`` (relaxation parameters) on different `Julia` workers for each index ``i``.

#Parallel PARSDMM
To activate this type of parallelism, we only need to set one flag (`options.parallel = true`):

```julia
options.parallel       = true

#if we change to parallel computions, we need to repeat the precomputations and distribution
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
```

#Multilevel Serial PARSDMM
First we need to decide on the number of levels and the coarsening factor.
```julia
options.parallel = false

#2 levels, the gird point spacing at level 2 is 3X that of the original (level 1) grid
n_levels 		  = 2 #integer
coarsening_factor = 3 #float
```

The multilevel version of PARSDMM has a slightly different setup. What we did before as part of `PARSDMM_precompute_distribute`, is now included as part of `setup_multi_level_PARSDMM`. The quantity `TD_OP_levels` is a vector of vectors of linear operators. Access linear operator number ``1`` on level ``2`` as `TD_OP_levels = TD_OP[2][1]`, so the first index refers to the level. 

```julia
#set up all required quantities for each level
(TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels) = setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
```

#Multilevel Parallel PARSDMM
The only difference from serial multilevel PARSDMM is the flag `options.parallel = true`.

```julia
#now use multi-level with parallel PARSDMM
options.parallel = true

#2 levels, the gird point spacing at level 2 is 3X that of the original (level 1) grid
n_levels=2
coarsening_factor=3

#set up all required quantities for each level
(TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels) = setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
```