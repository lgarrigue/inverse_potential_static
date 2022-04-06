include("misc/lobpcg.jl")
include("misc/misc.jl")
include("misc/plot_funs.jl")
using FFTW
using WriteVTK, Optim #, PlotlyJS
#pgfplotsx() # backend for tikz format export
#pyplot() # backend for ps format export
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra
using Random
#gr(show=false)

const mop = Manopt
const maf = Manifolds
const mab = ManifoldsBase

zeroar(p,complex=false) = zeros(complex ? ComplexF64 : Float64,Tuple(fill(p.Nx,p.dim)))
onear(p,complex=false)   = ones(complex ? ComplexF64 : Float64,Tuple(fill(p.Nx,p.dim)))


#CONTROL REPARTITION OF MANY-BODY STATES (is it really centered on the degenerate space ?)
#rep = repartition(E_many,configs)
# ET SORTIR UNE GRANDEUR QUI QUANTIFIERA LA DEGENERESCENCE CONTINUEMENT

mutable struct Params
	dim; Nx # number of discretization on one direction
	N # number of electrons
	kth # k th excited state, start from 1 (ground state)
	Nadd_1; Nadd_N

	pod

	x_axis_cart; x_axis_red; x_grid_cart; x_grid_red # Array of axis for abscisses for instance
	Nfull; dx; dvol; L # physical lattice size, [0,L]
	k_axis; k_grid; k2grid; k2lin
	
	gauge_vec

	### Max iterations
	max_iter_inv # maximum number of iterations for the inversion
	max_iter_pas
	max_iter_oda

	### Pas parameters
	pas
	pas_init
	learn_pas
	optim_pas
	optimize_pas_ρ_or_G

	### Temperature parameters
	deg_mult
	Temp
	cool_factor
	ΔE # energy cut-off for degeneracies (actual cut off is 2*ΔE)
	divi

	### Tolerances
	tol_lobpcg
	tol_oda
	tol_on_ρ

	### Plots
	Nx_plots
	x_axis_red_plots
	x_axis_cart_plots
	cutoff_plots
	function Params()
		new()
	end
end

function init_other_params(p)
	p.dx = p.L/p.Nx
	p.x_axis_cart = (0:p.Nx-1)*p.dx
	p.x_axis_red = (0:p.Nx-1)/p.Nx
	p.x_grid_cart = axis2grid(p.x_axis_cart,p)
	p.x_grid_red = axis2grid(p.x_axis_red,p)
	p.Nfull = p.Nx^p.dim
	p.dvol = p.dx^p.dim
	p.k_axis = fftfreq(p.Nx)*p.Nx
	p.k_grid = axis2grid(p.k_axis,p)
	p.k2grid = norm.(p.k_grid).^2
	p.k2lin = vcat(p.k2grid...) # kinetic energy operator
	p.ΔE = -1
	p.pas = p.pas_init
	p.gauge_vec = onear(p,false) # to fix the gauge
	p.x_axis_red_plots = (0:p.Nx_plots-1)/p.Nx_plots
	p.x_axis_cart_plots = (0:p.Nx_plots-1)*p.L/p.Nx_plots
end

######################## Solves one body

integral(ϕ,p) = p.dvol*sum(ϕ)
sca(ϕ,ψ,p) = p.dvol*ϕ⋅ψ
norm2(ϕ,p) = real(sca(ϕ,ϕ,p))
norms(ϕ,p) = sqrt(norm2(ϕ,p))
metrics(ρ,χ,p) = norms(sqrt.(ρ).-sqrt.(χ),p) # natural metrics on ρ
normalize(ρ,N,p) = N*ρ/integral(ρ,p) # normalizes to N, in L^1

cyclic_conv(a,b) = ifft(fft(a).*fft(b))

# Gives the action (hat{VX})_k in Fourier space. Vfour is the Fourier transform of V, Xfour is the Fourier transform of X
actionV(Vfour,p) = Xfour -> vcat(cyclic_conv(Vfour,Xfour)/length(Vfour)...)
actionH(actV,p) = X -> p.k2lin.*X .+ actV(Reshape(X,p)) # X = Xlin, v in direct space
Reshape(ψ,p) = reshape(ψ,Tuple(fill(p.Nx,p.dim)))

function solve_one_body(v,p)
	NtoComp = p.N + p.kth -1 + p.Nadd_1 # number of states to compute
	v_four = fft(v)
	V = actionV(v_four,p)
	H = actionH(V,p)
	(λs,ϕs,cv) = solve_lobpcg(H,p.Nfull,NtoComp,p.k2lin;tol=p.tol_lobpcg)
	ψs = []
	for i=1:length(ϕs)
		ψ = ifft(Reshape(ϕs[i],p))
		ψ = real.(ψ)
		ψ ./= norms(ψ,p)
		# s = sca(p.gauge_vec,ψ,p)
		# θ = acos(real(s/abs(s)))
		# ψ .*= cis(-im*θ)
		push!(ψs,ψ)
	end
	(λs,ψs)
end



#function reinitialize(p,p_init) # reinitializes some values of p from ones of p_init, to perform inversion of all times of a dynamic evolution
#p.
#end

######################## Others

function shift(f,d,a)
	if d==1
		return r -> f(x-a)
	elseif d==2
		return r -> f(r[1]-a,r[2]-a)
	elseif d==3
		return r -> f(r[1]-a,r[2]-a,r[3]-a)
	end
end

function step(a,p) # step function, with array a, for instance a = [0,3,-4,7,2] for d=1, or [[1,2],[3,4]] for d=2
	M = size(a,1)
	na = zeroar(p,false)
	J = Int(floor((p.Nx-1)/M))
	shift = p.dim == 1 ? 3 : 1
	if p.dim==1
		for x=1:p.Nx
			n = mod1(1+Int(floor((x-1)/J)),M)
			na[mod1(x+shift,p.Nx)] = a[n]
		end
	elseif p.dim==2
		for x=1:p.Nx
			for y=1:p.Nx
				n = mod1(1+Int(floor((x-1)/J)),M)
				m = mod1(1+Int(floor((y-1)/J)),M)
				na[mod1(x+shift,p.Nx),mod1(y+shift,p.Nx)] = a[n,m]
			end
		end
	end
	na
end

function density_of_config(orbs,conf,p) # density of the config (array of ints), takes all orbitals of the config
	ρ = zeroar(p,false)
	if conf == []
		return ρ
	end
	for i=1:length(conf)
		ρ .+= abs2.(orbs[conf[i]])
	end
	# px("NORM ints ",integral(ρ,p)," conf ",conf," norm ",norm2(orbs[1],p)," norm2 ",norm2(orbs[2],p))
	ρ
end

function regularizes(a,λ,p)
	b = similar(a)
	b .- integral(b,p)
	for i=1:length(a)
		x = a[i]
		b[i] = x≤-λ ? -λ : (x≥λ ? λ : x)
	end
	b
end

function invert_one_body_density(ρ,λ,p) # v = (Δ sqrt(ρ))/sqrt(ρ)
	#regu = x -> x>= ε ? x : (x <= -ε ? x : -ε)
	#regu = x -> x>= a ? a : x
	sr = sqrt.(ρ)
	del = Δper(sr,p.dx)
	regularizes(del./sr,λ,p)
end

mutable struct Config # many-body configurations
	conf # list of orbitals [2,4,5]
	energy # energy of the sum of orbitals
	kth # it is the k th energy state of the many-body Ham
end

function energy_of_config(E,conf) # energy of the config
	if conf == []
		return 0
	end
	energies = map(i -> E[i],conf) # just E[conf] ?
	return sum(energies)
end

function sort_many_body_energies(E,N,k,Nadd) # returns all the configurations with their energies, by increasing number of energy. add : additional orbitals taken into account
	add = k -1 + Nadd
	#println(add,length(E))
	@assert N+add ≤ length(E)
	all = collect(combinations(1:N+add,N)) # list of all combinations
	energies = map(conf -> energy_of_config(E,conf),all)
	sp = sortperm(energies)
	configs = []
	for i = 1:length(all)
		j = sp[i]
		C = Config(all[j],energies[j],i)
		push!(configs,C)
	end
	configs
end


function cut_energies_config(configs,k,ΔE) # takes a list of configurations having sorted energies, returns all the configurations having energy E such that ab(E-E^kth) \le delta, by increasing number of energy, and all the many-body labels
	n = length(configs)
	reference = configs[k].energy
	new_configs = []; manybody_labels = []
	enough_energies = true # will check whether we have enough energies
	for i=1:n
	c = configs[i]
		#println("{{{{ ",i," ",k," ",c.energy," ",reference," ",ΔE," result ",abs(c.energy - reference) ≤ ΔE)
		if abs(c.energy - reference) <= ΔE
			push!(new_configs,c)
			push!(manybody_labels,i)
			if i == n
				println("NOT ENOUGH ENERGIES !")
			end
		elseif i > k
			break
		end
	end
	@assert length(new_configs) >= 1
	(new_configs,manybody_labels)
end

# ======== Extract inner and outer orbitals
function possible_orbs(configs) # returns the list of all orbitals that appear in the configs
	l = []
	for i=1:length(configs)
		c = configs[i].conf
		for j=1:length(c)
			a = arglist(l,c[j])
			if a ==-1
				push!(l,c[j])
			end
		end
	end
	l
end

function is_in_all(a,configs) # checks if a is in all orbitals of the configs
	for i=1:length(configs)
		c = configs[i].conf
		if arglist(c,a) == -1
			return false
		end
	end
	true
end

# l and s are arrays, and we substract the elements of s from the ones of l, returns {l}\{s}
set_difference(l,s) = filter!(e->e∉s,copy(l)) # copy, otherwise it modifies it !

function extract_outer_orbitals(configs) # takes a config and returns the orbitals which are in common, and the outer configs. [1,2,3,5], [1,2,8,9] will give [1,2],([3,5],[8,9]) (does not give a config, but an array of orbs labels)
	possibles = possible_orbs(configs)
	inner_orbs = []
	for i=1:length(possibles)
		if is_in_all(possibles[i],configs)
			push!(inner_orbs,possibles[i])
		end
	end
	outer_configs = []
	for i=1:length(configs)
		new_conf = set_difference(configs[i].conf,inner_orbs)
		push!(outer_configs,new_conf)
	end
	(inner_orbs,outer_configs)
end

# ========== Prints configs
function print_configs(configs)
	if length(configs)==0
		println("EMPTY CONFIG")
	else
		for i=1:length(configs)
			c = configs[i]
			println(c.kth," th config is ",c.conf,", en ",c.energy)
		end
	end
end

function arglist(l,e) # checks if e is in l, returns -1 if not, returns the label otherwise
	a = findall(x->x==e,l)
	if length(a) == 0
		return -1
	end
	return a[1]
end

function one_diff(a,b,prt=false) # a and b are arrays of integers. Does not give -1 iff a and b have only one orbital difference, in which case it gives the two labels
	c = copy(a); d = copy(b)
	lastlength = length(c)+1
	constant = false
	while lastlength!=length(c) && length(c) >= 1
		#println(c,d,"\n\n")
		for i = 1:length(c)
			m = arglist(d,c[i])
			if !(m==-1)
				deleteat!(c,i)
				deleteat!(d,m)
				break
			end
			#if length(c) == lastlength
			#	constant = true
			#end
			lastlength = length(c)
		end
	end
	res = 0
	if !(length(c)==1)
		res = -1
	else
		res = (c[1],d[1])
	end
	if prt
		println("Compared ",a," with ",b," returns ",res)
	end
	res
end

mutable struct MixedState
	S # n times n real symmetric matrix
	n
	energies # the n energies of the pure states
end

#function energy(S) # energy of a mixed state

#end

function confining_potential(p,fraction=20,heigth=1e3)
	f(x,y) = 0
	if p.dim == 1
		f(x) = (x ≤ 1/fraction || x ≥ 1-1/fraction) ? heigth : 0
	elseif p.dim == 2
		f(x,y) = (x ≤ 1/fraction || x ≥ 1-1/fraction || y ≤ 1/fraction || y ≥ 1-1/fraction) ? heigth : 0
	elseif p.dim == 3
		f(x,y,z) = (x ≤ 1/fraction || x ≥ 1-1/fraction || y ≤ 1/fraction || y ≥ 1-1/fraction || z ≤ 1/fraction || z ≥ 1-1/fraction) ? heigth : 0
	end
	eval_f(f,p)
end

function zero_borders(f,fraction=100) # transforms it to the same but with zero at the border
	nf(x) = (x ≥ 1-1/fraction || x ≤ 1/fraction) ? 0 : f(x)
	if p.dim==2
		nf(x,y) = (x ≥ 1-1/fraction || x ≤ 1/fraction || y ≥ 1-1/fraction || y ≤ 1/fraction) ? 0 : f(x,y)
	elseif p.dim==3
		nf(x,y,z) = (x ≥ 1-1/fraction || x ≤ 1/fraction || y ≥ 1-1/fraction || y ≤ 1/fraction || z ≥ 1-1/fraction || z ≤ 1/fraction) ? 0 : f(x,y,z)
	end
	nf
end

# ============= ODA

function overlaps_densities(outer_configs,orbs,pa) # ρ_IJ = (N * int_(N_out-1) psi_I psi_J) overlap functions
	n = length(outer_configs)
	zero = zeroar(pa)
	M = [zero for i=1:n, j=1:n] # matrix of functions
	for i=1:n
		oc = outer_configs[i]
		for j=1:n
			if i==j
				for k=1:length(oc)
					M[i,i] += orbs[oc[k]] .^2
				end
			else
				diff = one_diff(oc,outer_configs[j])
				if diff != -1
					M[i,j] = orbs[diff[1]] .* orbs[diff[2]]
				end
			end
		end
	end
	M
end

function rotate_tensor(n,k,proj,ρ_IJ,pa) # proj is a n x k matrix, rotates and reduces ρ_IJ
	zero = zeroar(pa)
	M = [zero for i=1:n, j=1:n]
	for i=1:n
		for j=1:n
			for m=1:k
				for h=1:k
					M[i,j] += conj(proj[i,m])*proj[j,h]*ρ_IJ[m,h]
				end
			end
		end
	end
	M
end

function rotate_integrals(n,k,proj,integral,pa)
	M = zeros(ComplexF64,n,n)
	for I=1:n
		for J=1:n
			for Ip=1:k
				for Jp=1:k
					M[I,J] += conj(proj[I,Ip])*proj[J,Jp]*integral[Ip,Jp]
				end
			end
		end
	end
	M

end

# function double_rotate_integrals2(n,k,proj,integrals,pa) # proj is a n x k matrix, rotates and reduces int ρ_IJ ρ_KL
	# M = zeros(ComplexF64,n,n,n,n)
	# for I=1:n
		# for J=1:n
			# for K=1:n
				# for L=1:n
					# for Ip=1:k
						# for Jp=1:k
							# for Kp=1:k
								# for Lp=1:k
									# M[I,J,K,L] += conj(proj[I,Ip])*proj[J,Jp]*conj(proj[K,Kp])*proj[L,Lp]*integrals[Ip,Jp,Kp,Lp]
								# end
							# end
						# end
					# end
				# end
			# end
		# end
	# end
	# M
# end

# proj is a n x k matrix, rotates and reduces int ρ_IJ ρ_KL
double_rotate_integrals(n,k,proj,integrals,pa) = [sum([conj(proj[I,Ip])*proj[J,Jp]*conj(proj[K,Kp])*proj[L,Lp]*integrals[Ip,Jp,Kp,Lp] for Ip=1:k,Jp=1:k,Kp=1:k,Lp=1:k]) for I=1:n,J=1:n,K=1:n,L=1:n]

function nabla_E(outer_configs,E,orbs,pot,p) # matrix ∇E at M (does not depend on M)
	n = length(outer_configs)
	S = zeros(ComplexF64,(n,n))
	for i=1:n
		oc = outer_configs[i]
		for j=1:n
			if i==j
				S[i,i] = energy_of_config(E,oc)
			else
				diff = one_diff(oc,outer_configs[j])
				if diff != -1
					o = orbs[diff[1]]
					S[i,j] = sca(Δ(o,p.dx) + pot.* o,orbs[diff[2]],p)
				end
			end
		end
	end
	S
end

function f_target(Γ,datas,p,δ=1)
	dpr = Γ⋅datas.∇E
	argexp = real(datas.diff_en + dpr)
	expo = exp(δ*argexp^2/(datas.Temp^2))
	real(expo*big_factor(Γ,datas))
end

function big_factor(Γ,datas)
	m = datas.m
	datas.int_ρ_diff + 2* Γ⋅datas.diff_int_KL + sum([Γ[i,j]*Γ[k,l]*datas.tensor[i,j,k,l] for i=1:m, j=1:m, k=1:m, l=1:m])
end


function nabla_f(Γ,outer_configs,datas,p,δ,one_diff_accelerate=true) # matrix ∇f_target at M (does not depend on M), one_diff_accelerate = true when on the natural basis
# Γ is the symmetric matrix of coefficients
	n = datas.m
	S = zeros(ComplexF64,n,n)
	argexp = datas.diff_en + Γ⋅datas.∇E
	expo = min(exp(real(δ*argexp^2/datas.Temp^2)),1e10) # OTHERWISE, THIS BRINGS Inf sometimes
	bigfactor = δ*(1/datas.Temp^2)* argexp* big_factor(Γ,datas)
	for i=1:n
		for j=1:n
			#if (one_diff_accelerate && (i==j || one_diff(outer_configs[i],outer_configs[j]) != -1)) || (!one_diff_accelerate)
			S[i,j] = 2*datas.diff_int_KL[i,j]+2*sum([Γ[k,l]*datas.tensor[i,j,k,l] for k=1:n, l=1:n]) + datas.∇E[i,j] * bigfactor
			#end
		end
	end
	#px("temp ",sum(abs.(datas.∇E)))
	expo*S
end

mutable struct DatasOpt
	m
	diff_en
	ρ_diff
	ρ_IJ
	int_ρ_diff
	diff_int_KL
	tensor
	∇E
	Temp
end

function density_of_Γ(Γ,outer_configs,datas,p) # density of the outer config
	ρ = zeroar(p,false)
	n = datas.m
	for i=1:n
		for j=1:n
			#if i==j || one_diff(outer_configs[i],outer_configs[j]) != -1
			ρ += Γ[i,j]*datas.ρ_IJ[i,j]
			#end
		end
	end
	@assert integral(abs.(imag.(ρ)),p) <= 0.05
	real.(ρ)
end

function build_data(E_kth,tρ,pot,inner_orbs,outer_configs,E,orbs,p)
	# E_kth is the energy of the kth many-body level
	# E are the energies of the one-body
	# tρ is the target density
	m = length(outer_configs)
	diff_en = energy_of_config(E,inner_orbs) - E_kth # e_in - E^k(v)
	ρ_diff = density_of_config(orbs,inner_orbs,p)-tρ # ρ_in - tρ
	ρ_IJ = overlaps_densities(outer_configs,orbs,p) # ρ_IJ
	int_ρ_diff = integral(ρ_diff.^2,p) # int (ρ_in-tρ)^2
	diff_int_KL = [integral(ρ_diff.*ρ_IJ[i,j],p) for i=1:m, j=1:m] # matrix int (ρ_in - tρ) ρ_KL
	tensor = [integral(ρ_IJ[i,j] .* ρ_IJ[k,l],p) for i=1:m, j=1:m, k=1:m, l=1:m] # tensor int ρ_IJ ρ_KL
	∇E = nabla_E(outer_configs,E,orbs,pot,p) # matrix e_{IJ}
	DatasOpt(m,diff_en,ρ_diff,ρ_IJ,int_ρ_diff,diff_int_KL,tensor,∇E,p.Temp)
end

function build_data_rotated(n,k,proj,data,p)
	@assert k == data.m
	proj = proj'
	ρ_IJ = rotate_tensor(n,k,proj,data.ρ_IJ,p) # proj is a n x k matrix, rotates and reduces ρ_IJ
	diff_int_KL = rotate_integrals(n,k,proj,data.diff_int_KL,p)
	tensor = double_rotate_integrals(n,k,proj,data.tensor,p)
	∇E = rotate_integrals(n,k,proj,data.∇E,p)
	#=function ssum(a)
		sum(abs.(a))
	end
	px("proj ",ssum(proj))
	#px("nablaE ",ssum(∇E)," tensor ",ssum(tensor)," diff_intkl ",ssum(diff_int_KL)," rhos ",sum( [sum(abs.(ρ_IJ[i,j])) for i=1:n, j=1:n]))
	=#
	DatasOpt(n,data.diff_en,data.ρ_diff,ρ_IJ,data.int_ρ_diff,diff_int_KL,tensor,∇E,p.Temp)
end

# ===== Min on wavefunctions

function density_pure(tρ,inner_orbs,outer_configs,E,orbs,p) # finds the closest density to the target one, in the degenerate density space
	datas = build_data(0,tρ,tρ,inner_orbs,outer_configs,E,orbs,p)
	M = mop.Grassmann(datas.m,1,ℂ)
	# C has to be a column vector
	cost = C -> real(big_factor(real.(C*C'),datas))
	#=
	if false
		# Random search
		a = []; b=[]
		for i=1:max_iter
			Copt = mop.random_point(M)
			push!(a,Copt)
			push!(b,cost(Copt))
		end
		i = argmin(b)
		Copt = a[i]
	else
	=#
	transport = ProjectionTransport()
	Copt = rand(p.Nx) #mop.particle_swarm(M, cost; vector_transport_method = transport,stopping_criterion=StopAfterIteration(max_iter))
	ρ_opt = density_of_Γ(real.(Copt*Copt'),outer_configs,datas,p)
	den = ρ_opt + density_of_config(orbs,inner_orbs,p)
	@assert integral(abs.(imag.(den)),p) <= 0.05
	real.(den)
end


function ODA(datas,outer_configs,p,min_or_max_or_middle="min") # if min or max or neither min nor max (middle)
	δ = min_or_max_or_middle == "min" ? 1 : (min_or_max_or_middle == "max" ? -1 : 0)
	Γ_before = zeros(ComplexF64,datas.m,datas.m)
	Γ = (1/datas.m)*id_mat(datas.m,true) # tr Γ = 1, Yes complex true
	niter = 0
	while norm(Γ-Γ_before) > p.tol_oda && p.max_iter_oda -1 >= niter
		niter += 1
		∇f = nabla_f(Γ,outer_configs,datas,p,δ)
		#px("non herm part ",sum(abs.(∇f-∇f')))
		#@assert sum(abs.(∇f-∇f')) <= 0.05
		∇f = (∇f+∇f')./2 # Hermitian part
		v = eigen(∇f).vectors[:,1] # smallest eigenvector # THE PROBLEM IS HERE
		P = v*v' # its projector
		res = optimize(t->δ*(f_target((1-t)*Γ + t* P,datas,p,δ)), 0, 1; rel_tol=1e-2)
		t_opt = Optim.minimizer(res)
		Γ_before = Γ
		Γ = (1-t_opt) * Γ + t_opt * P
		Γ = (Γ+Γ')./2 # Hermitian part
		#px("res oda ",sum(abs.(Γ)))
	end
	#px("ODA ",niter," iterations")
	#px(" antisym ",sum(abs.(Γ-transpose(Γ))))

	# Verify mass
	if abs(tr(Γ)-1) >= 0.05
		px("ERROR OUTPUT ODA, MASS ",tr(Γ))
	end
	#Γ/real(tr(Γ))
	Γ
end

mutable struct Results
	L2dists_ρ
	L2dists_v
	degeneracies
	pas_list
	niter_list
	dist_pure
	G
	Results() = new([],[],[],[],[],[],[])
end

function plot_energies(E,ΔE,filled=[],title="") # filled = array of orbitals on which we put a mark
	plo = Plots.plot(legend=false,title=title,xaxis=([], false),grid=false)
	n = length(E)
	indent = 0
	interval = (E[n]-E[1])/20
	for i=1:n
		plot!(plo,[indent,indent+1],[E[i],E[i]])
		if i in filled
			plot!(plo,[indent+1/2,indent+1/2],[E[i]-interval,E[i]+interval],color = :black)
		end
		if i <= n-1
			if E[i+1]-E[i] <= ΔE
				indent += 1.5
			else
				indent = 0
			end
		end
	end
	plo
end

function plot_results(res)
	cat = vcat(res.L2dists_ρ,res.degeneracies,res.pas_list,res.niter_list,res.G)
	maxi = 3 # maximum(cat)
	plo = Plots.plot(ylims = (0,maxi),grid=false,legend=:topleft,legendfontsize=8)
	axis = [i for i=1:length(res.L2dists_ρ)]
	plot!(plo,axis,res.L2dists_ρ,label="L2 dist mixed")#/maximum(res.L2dists_ρ))
	plot!(plo,axis,res.dist_pure,label="L2 dist pure")
	plot!(plo,axis,res.degeneracies,seriestype = :scatter,label="deg")
	plot!(plo,axis,res.pas_list,seriestype = :scatter,label="pas")
	plot!(plo,axis,res.G,label="G(v)")
	#plot!(plo,axis,res.niter_list,seriestype = :scatter)
	plo
end

function new_result(res,er_ρ,er_v,deg,pas,niter,dist_pure,G)
	push!(res.L2dists_ρ,er_ρ)
	push!(res.L2dists_v,er_v)
	push!(res.dist_pure,dist_pure)
	push!(res.G,G)
	push!(res.degeneracies,deg)
	push!(res.pas_list,pas)
	push!(res.niter_list,niter)
end

function gaussian_orbs(p) # N gaussians orbs,
	N = p.N
	σ = p.L/(5*N)
	ecart = 2*p.L/N
	gauss(x) = (2*π*σ)^(-p.dim/2) * exp(-x^2/(2*σ^2))
	centers = [-p.L + ecart/2 + ecart*n for n=0:N-1]
	function f(x,b,ce,gau)
	#px("=========== ",b, " ",abs.(x.-centers)," ",argmin(abs.(x.-centers)))
		if argmin(abs.(x.-centers))==b
			return gau(x-centers[b])
		else
			return 0
		end
	end
	l = []
	for n=1:N
		a = eval_f(x->f(x,n,centers,gauss),p)
		push!(l,normalizeSq(a,p))
	end
	l
end



#=
function ρ_cross(a,b,orbs,p) # a and b are configs
	ρ = zeroar(p)
	if a==b
		for k=1:length(a)
			ρ += orbs[a[k]] .^2
		end
	else
		diff = one_diff(a,b)
		if diff != -1
			ρ = orbs[diff[1]] .* orbs[diff[2]]
		end
	end
	ρ
end

function AB_deg2(u,dcos,dsin,p)
	(int((1/2)*dcos.*u,p) , int(dsin.*u,p))
end

function u_of_theta(θ,c,dcos,dsin,p)
	tot = c + dcos * cos(2*θ) /2 + dsin * sin(2*θ)
	tot/norm2(tot,p)
end

function target_deg2(θ,c,dcos,dsin,ε,p)
	u = u_of_theta(θ,c,dcos,dsin,p)
	(A,B) = AB_deg2(u,dcos,dsin,p)
	int(c.*u,p) - ε*sqrt(A^2 + B^2)
end

function degeneracy2(ρin,tρ,outer_confs,orbs,p,ε=1) # in case the degeneracy is 2, to be added for the optimal direction
	c1 = outer_confs[1]
	c2 = outer_confs[2]
	ρ1 = density_of_config(orbs,c1,p)
	ρ2 = density_of_config(orbs,c2,p)
	c =  ρin - tρ + (ρ1+ρ2)/2
	o1 = orbs[c1][1]
	o2 = orbs[c2][1]
	#px(" mmm ",int(o1.*o2,p))
	# GROS PB : o1 et o2 SONT PROPORTIONNELLES !!!
	
	#display(plArs([o1,o2],p))
	px("int ",int(o1.*o2,p))
	sleep(3)
	dcos = ρ1-ρ2
	dsin = ρ_cross(c1,c2,orbs,p)
	#px("ici ",int(abs.(dsin),p))

	# Optimizes over directions
	res = optimize(θ->-1*target_deg2(θ,c,dcos,dsin,ε,p), 0, 2*π; rel_tol=1e-3) # MAXIMIZE HERE !
	θ_opt = Optim.minimizer(res)
	u_opt = u_of_theta(θ_opt,c,dcos,dsin,p)

	# Computes the density
	(A,B) = AB_deg2(u_opt,dcos,dsin,p)
	ρ_opt = ρin + (ρ1+ρ2)/2 - ε*(A*dcos + 2*B*dsin)/(2*sqrt(A^2+B^2))

	px("shld be 0 ",norm2(u_opt - (ρ_opt-tρ)/norm2(ρ_opt-tρ,p)  ,p))
	ρ_opt
end
=#

eval_f(f,p,complex=false) = eval_f_raw(f,p.dim,p.Nx,complex)

function axis2grid(ax,p)
	grid = []
	if p.dim == 1
		return ax
	elseif p.dim == 2
		return [[ax[i],ax[j]] for i=1:p.Nx, j=1:p.Nx]
	elseif p.dim == 3
		return [[ax[i],ax[j],ax[l]] for i=1:p.Nx, j=1:p.Nx, l=1:p.Nx]
	end
end

# from a, builds the Fourier transform, transform it as a function, evaluates it on a more precise grid
function interpolate(a,p)
	fa = fft(a)
	f = fft_to_fun_raw(fa,p.dim,p.Nx,p.k_axis,p.cutoff_plots)
	b = eval_f_raw(f,p.dim,p.Nx_plots,false)/(p.Nx^p.dim)
	# px("Test norm ",sum(b)*((p.L/p.Nx_plots)^p.dim))
	b
end
