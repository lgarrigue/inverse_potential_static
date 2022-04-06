# Main functions

using DFTK, LinearAlgebra, Combinatorics, Random #PyCall, DSP, 
using Manopt, Manifolds, ManifoldsBase

using Plots#, PyPlot, GR; pyplot()

include("structures.jl")

const mop = Manopt
const maf = Manifolds
const mab = ManifoldsBase

mutable struct Solution
	ρ
	pot
	outer_configs
	all_configs
	E
	E_many_body
	inner_orbs
	orbs
	ρ_pure
end

G(tρ,sol,p) = sol.all_configs[p.kth].energy - integral(sol.pot.*tρ,p) # target function we maximize

function solve_all(pot,tρ,p) # computes the steepest ascent direction (actually the one of k=m_k)
	# px("NORM tρ ",integral(tρ,p)) # ok
	
	# Solve the one-body problem, with excited states
	(E,orbs) = solve_one_body(pot,p) # one-body
	#p.Nadd_N = maximum(p.Nadd_N,max_level(E) - (p.N+p.kth-1)) # to know how many one-body orbitals to take into account in the many-body energies
	# px("NORM orbs ",integral(abs2.(orbs[1]),p)) # ok

	# Creates and sorts the many-body configurations, by increasing energies
	all_configs = sort_many_body_energies(E,p.N,p.kth,p.Nadd_N)
	E_many_body = map(c->c.energy,all_configs)
	Ekth = all_configs[p.kth].energy

	# Defines the many-body energy threshold and the "temperature"
	if p.ΔE == -1
		p.ΔE = (E[p.N+p.kth-1]-E[1])/(p.divi*(p.N+p.kth)) # do not do so with many-body energy, they can be equal
		p.Temp = p.ΔE/p.deg_mult
	end

	# Drop the configurations which are not in the energy band, gives the indices of the labels of the degenerate configs
	(configs,mb_deg_levels) = cut_energies_config(all_configs,p.kth,p.ΔE)

	# Among those configurations, finds if some orbitals are common, split between inner orbitals and outer configurations
	(inner_orbs,outer_configs) = extract_outer_orbitals(configs)
	ρ = density_of_config(orbs,inner_orbs,p)
	# px("NORM rho ",integral(ρ,p)) # not ok
	deg = length(outer_configs)

	#if deg == 2 && false
		#ρ = degeneracy2(ρ,tρ,outer_configs,orbs,p,1)
	
	dim = maximum(mb_deg_levels)-p.kth+1
	datas = build_data(Ekth,tρ,pot,inner_orbs,outer_configs,E,orbs,p)
	if deg == 1
		a = 1
	elseif true #minimum(mb_deg_levels) == p.kth
		Γopt = ODA(datas,outer_configs,p,"min")
		ρ += density_of_Γ(Γopt,outer_configs,datas,p)
	elseif maximum(mb_deg_levels) == p.kth
		Γopt = ODA(datas,outer_configs,p,"max")
		ρ += density_of_Γ(Γopt,outer_configs,datas,p)
	else
		# Computes the dimension of the space Q (the max)
		Grass = mop.Grassmann(deg,dim,ℂ) # matrices of size deg x dim

		function cost(proj)
			#=
			if isnan(sum(abs.(proj)))
				return -100000000
			end
			=#
			datas_rot = build_data_rotated(dim,deg,proj,datas,p)
			#px("nabla ",sum(abs.(datas_rot.∇E)))
			Γ = ODA(datas_rot,outer_configs,p,"min")
			res = f_target(Γ,datas_rot,p,1)
			res
		end

		# Extracts optimal density
		# if false
		# proj_opt = mop.particle_swarm(Grass, proj -> -1*cost(proj,outer_configs); vector_transport_method = ProjectionTransport(),stopping_criterion=StopAfterIteration(5),n=3)
		# else
		a = []; b=[]
		#niter = manifold_dimension(Grass)==0 ? 1 : 10
		for i=1:10
			pro = mop.random_point(Grass)
			px("projection ",pro," dim ",)
			push!(a,pro)
			push!(b,cost(pro))
		end

		i = argmax(b)
		proj_opt = a[i]

		push!(b,cost([1/dim for i=1:dim, j=1:dim]))
		px(" lala ",b)
		datas_rot = build_data_rotated(dim,deg,proj_opt,datas,p)
		px("nablas ",sum(abs.(datas_rot.∇E))," et true ",sum(abs.(datas.∇E))," tensors ",sum(abs.(datas_rot.tensor))," et true ",sum(abs.(datas.tensor)))
		# end
		datas_rot_opt = build_data_rotated(dim,deg,proj_opt,datas,p)
		Γopt = ODA(datas_rot_opt,outer_configs,p,"min")
		px("DIM ",dim," x ",deg)
		ρ += density_of_Γ(Γopt,outer_configs,datas_rot_opt,p)
		#dist = norm2(dens-tρ,p)
	end

	# Verifies that the solution has the right mass
	if abs(integral(ρ,p)-p.N)>=0.05
		px("NOT RIGHT MASS MIXED ",integral(ρ,p))
	end
	Solution(ρ,pot,outer_configs,all_configs,E,E_many_body,inner_orbs,orbs,nothing)
end

function optimize_pas(tρ,pot,p,accelerate,last_sol)
	best_direction = zeroar(p,false)
	go_on = true
	accelerator = accelerate ? 5 : 1
	initial_pas = p.pas*accelerator
	pas = initial_pas
	pas_list = [float(pas)]
	dist_list = []; G_list = []; sol_list = []
	l = 1
	inc_dec = "increase"

	# Remember last solution if exists
	sol = 0
	if last_sol == 0
		sol = solve_all(pot,tρ,p) # gives the best direction
	else
		sol = last_sol
	end
	best_direction = (sol.ρ-tρ) /norm2(sol.ρ-tρ,p)
	
	# If constant pas
	if !p.optim_pas
		pot_test = pot + p.pas*best_direction
		sol = solve_all(pot_test,tρ,p)
		ρ_pure = density_pure(tρ,sol.inner_orbs,sol.outer_configs,sol.E,sol.orbs,p)
		
		return (p.pas,G(tρ,sol,p),sol,1,ρ_pure)
	end
	while go_on
		#px(outer_configs_test)
		pot_test = pot + pas*best_direction
		sol = solve_all(pot_test,tρ,p)

		# Optimize wrt densities or maximizing function
		dist_test = norm2(sol.ρ-tρ,p)
		G_val = G(tρ,sol,p)

		push!(sol_list,sol)
		push!(dist_list,dist_test)
		push!(G_list,G_val)

		compare_val = p.optimize_pas_ρ_or_G == "ρ" ? dist_test : -G_val
		compare_list = p.optimize_pas_ρ_or_G == "ρ" ? dist_list : -1*G_list

		if l==2 && compare_val > compare_list[1]
			inc_dec = "decrease"
			pas = initial_pas
		end

		if l==1
			pas = pas*1.2
			push!(pas_list,pas)
		elseif (compare_val > compare_list[length(compare_list)-1] && l >= 2 && inc_dec == "increase") || (compare_val > (l==3 ? compare_list[1] : compare_list[length(compare_list)-1]) && l >= 3 && inc_dec == "decrease")
			go_on = false
		else
			ratio = inc_dec == "increase" ? 1.2 : 0.7
			pas = pas*ratio
			push!(pas_list,pas)
		end

		#px(" compare_list ",inc_dec," ",compare_list)
		if l >= p.max_iter_pas
		#if (l >= p.max_iter_pas && inc_dec == "decrease") || (l >= p.max_iter_pas && inc_dec == "increase")
			go_on = false
		end
		#px("int ",integral(abs.(pas*best_direction),p))
		l += 1
		if false
			go_on = false
		end
	end
	b = argmin(p.optimize_pas_ρ_or_G == "ρ" ? dist_list : -1*G_list)
	if p.learn_pas
		p.pas = pas_list[b]
	else
		p.pas = p.pas_init
	end

	sol = sol_list[b]
	ρ_pure = density_pure(tρ,sol.inner_orbs,sol.outer_configs,sol.E,sol.orbs,p)
	ρ_pure = normalize(ρ_pure,p.N,p)
	sol.ρ_pure = ρ_pure
	(pas_list[b],G_list[b],sol,l,ρ_pure)
end

function verify_mass(f,mass,err,p)
	erro = mass-integral(f,p)
	if erro >= err
		px("PROBLEM IN MASS = ", erro," >= ",err)
	end
end

function inverse_pot(tρ,p;plot_infos=false,init_v=-1,tv=0)
	pot = init_v==-1 ? invert_one_body_density(tρ,50,p) : (init_v==0 ? zeroar(p,false) : init_v)
	save_plot([pot,tρ],p,"init_pot_and_tρ")
	if plot_infos
		px("Start solving")
	end
	res = Results()
	j = 1
	sol = 0
	criterion = true
	while criterion
		if plot_infos
			px("Step ",j," ")
		end

		(pas_opt,G_opt,sol,niter,ρ_pure) = optimize_pas(tρ,pot,p,j==1,sol)
		pot = sol.pot
		pot = pot .- minimum(pot) # have the minimum of the potential to zero
		if p.pod=="pot"
			pot .+= onear(p) * mean_middle(tv.-pot,p,5)
			#pot += onear(p) * integral(tv-pot,p)/integral(onear(p),p)
		end

		verify_mass(ρ_pure,p.N,0.05,p)

		L2_dist_ρ = norms(sol.ρ-tρ,p)
		# L2_dist_ρ = metrics(sol.ρ,tρ,p)
		log_L2_dist_ρ = -log10(L2_dist_ρ)
		dist_pure= norms(ρ_pure-tρ,p)
		log_dist_pure= -log10(dist_pure)
		L2_dist_v = p.pod=="pot" ? norms(pot-tv,p) : 0

		deg = length(sol.outer_configs)

		p.learn_pas = true #(L2_dist_ρ/p.N <= 0.005)

		# ========== Analysis ==========
		new_result(res,log_L2_dist_ρ,L2_dist_v,deg,pas_opt,niter,log_dist_pure,G_opt/10)

		# ========== Plot ==========
		# px("N-body degeneracy: ",deg,", error ",log_L2_dist_ρ)
		title = "Step "*string(j)*", err "*float2str(log_L2_dist_ρ)*", max "*float2str(maximum(res.L2dists_ρ))*" max dpure "*float2str(maximum(res.dist_pure))*"\n, N-deg "*string(deg)*", N "*string(p.N)*", k "*string(p.kth)*",\n dim "*string(p.dim)*", T "*string(p.Temp)*(false ? ", error potentials "* string(L2_dist_v) : "")
		title_plot = Plots.plot(title=string(repeat(' ',20),title),grid=false,showaxis=false,axis=([], false),titlefontsize=9,titlefonthalign=:center,bottom_margin=-190Plots.px)
		#Dplot = p.dim==2 ? plot2d(tρ-sol.ρ) : plArs([tρ,sol.ρ],p)

		Eplot_1 = plot_energies(sol.E,p.ΔE,sol.all_configs[p.kth].conf) #,"\\sigma (-\\Delta+v)")
		Eplot_N = plot_energies(sol.E_many_body,p.ΔE,[p.kth])#,"\\sigma (-\\Delta_N+\\Sigma v)")
		pot_to_plot = max.(min.(pot,10000),-1000)
		pot_plot = p.pod=="den" ? [pot_to_plot] : [pot_to_plot,tv] 
		#Dplot = plArs(vcat([tρ,sol.ρ,ρ_pure]*10,pot_plot),p,"",[true,false,false,false,false],[:blue,:blue,:cyan,:red,:red],["tden","rho","rhopure","pot"],0,30)
		Dplot = plArs(vcat([tρ,sol.ρ],pot_plot),p,"",[true,false,false,true],[:blue,:blue,:red,:red],["tρ","ρ","v","tv"])
		res_plot = plot_results(res); general_plot = 0
		save_fig = false
		size = (1800,1000)
		ts = 2
		diffρ = log10.(abs.(sqrt.(tρ).-sqrt.(sol.ρ)))
		if p.dim == 1
			l = @layout [Plots.grid(1,1); Plots.grid(1,3); Plots.grid(1,3)]
			diffs = [diffρ]
			if p.pod=="pot"
				diffpot = log10.(abs.(tv-pot))
				push!(diffs,diffpot)
			end
			diffs_plot = plArs(diffs,p,"",[false,false],[:blue,:red])
			freqs_plot = plArs([sqrt.(abs.(fft(sol.ρ)))],p,"",[false],[:blue])
			general_plot = Plots.plot(title_plot,Eplot_1,Eplot_N,res_plot,Dplot,diffs_plot,freqs_plot,layout=l,thickness_scaling=ts,size=size)
		elseif p.dim == 2
			dρ_plot = plot2d(diffρ,[:black,:green,:yellow, :red])
			pot_plot = plot2d(pot_to_plot)
			tρ_plot = plot2d(tρ)
			ρ_plot = plot2d(sol.ρ)
			freqs_plot = plot2d(sqrt.(abs.(fft(sol.ρ))))
			dρ_plot = plot2d(diffρ)
			l = @layout [Plots.grid(1,1); Plots.grid(1,3); Plots.grid(1,3)]
			if p.pod=="den"
				#l = @layout [grid(1,1); grid(1,3); grid(1,3)]
				general_plot = Plots.plot(tρ_plot)#,dρ_plot,pot_plot,layout=l,titlefontsize=9,thickness_scaling=0.5)
				general_plot = Plots.plot(title_plot,Eplot_1,Eplot_N,res_plot,tρ_plot,dρ_plot,ρ_plot,pot_plot,freqs_plot,dρ_plot,thickness_scaling=1,size=(1000,800))
			else
				dpot_plot = plot2d(log10.(abs.(tv-pot)),[:black,:green,:yellow, :red])
				general_plot = Plots.plot(title_plot,Eplot_1,Eplot_N,res_plot,tρ_plot,pot_plot,freqs_plot,dρ_plot,titlefontsize=9,thickness_scaling=ts,size=size)
			end
		elseif p.dim == 3
			l = @layout [grid(1,1); grid(1,3); grid(1,1)]
			general_plot = Plots.plot(title_plot,Eplot_1,Eplot_N,res_plot,Dplot,layout=l,titlefontsize=9,thickness_scaling=ts,size=size)
		end
		if j ≤ 2 || mod(j,30)==0
			path = "intermediate_plots/"
			if !isdir(path) mkdir(path) end
			Plots.savefig(general_plot,string(path,"step",j,".png"))
		end
		if true
			if p.dim == 2 && false
				map(plo -> plot!(plo,tickfontsize=20,size=(400,360)),[tρ_plot,dρ_plot,pot_plot])
				savefig(dρ_plot,"plots/errDen.pdf")
				if p.pod=="pot" 
					savefig(diffspot,"plots/errPot.pdf")
				end
				savefig(tρ_plot,"plots/2dTDen.pdf")
				savefig(pot_plot,"plots/2dPot.pdf")
			elseif p.dim == 3
				save3d(tρ,p,"tden","Target density")
				save3d(pot,p,"pot","Potential")
				save3d(log10.(abs.(tρ-sol.ρ)),p,"logdiff","log10 |rho_n-rho|")
			end
		end

		# savefig(diffplot,"plots/2d.tikz")

		# ========== Cools down =========
		if mod(j,10)==0
			p.ΔE = p.ΔE/p.cool_factor
			p.Temp = p.ΔE/p.deg_mult
		end

		# ========== Plots for the article ==========
		if j ≥ p.max_iter_inv - 10 && p.dim==2
			scale_v = 10
			co = [:green,:yellow,:red]
			ts = 1.5
			p_ρ = Plots.heatmap(p.x_axis_cart,p.x_axis_cart,     tρ,grid=false,c=cgrad(co),thickness_scaling=ts,aspect_ratio=:equal,xlims=extrema(p.x_axis_cart),ylims=extrema(p.x_axis_cart),clims=(0,maximum(tρ)))
			p_v = Plots.heatmap(p.x_axis_cart,p.x_axis_cart,sol.pot,grid=false,c=cgrad(co),thickness_scaling=ts,aspect_ratio=:equal,xlims=extrema(p.x_axis_cart),ylims=extrema(p.x_axis_cart))
			Plots.savefig(p_ρ,string("plots/den",j,".png"))
			Plots.savefig(p_v,string("plots/pot",j,".png"))
		end

		# ========== Criterion ==========
		if L2_dist_ρ/p.N ≤ p.tol_on_ρ || j ≥ p.max_iter_inv
			criterion = false
			Plots.savefig(general_plot,string("final_plot.png"))
			return sol
		end
		j += 1
	end
	sol
end
