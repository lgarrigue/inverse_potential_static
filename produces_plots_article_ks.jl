using Plots, LaTeXStrings#, PyPlot, GR; pyplot()

include("structures.jl")
include("high_level_functions.jl")

function init_p()
	p = Params()

	### Space and configuration
	p.dim = 1; p.Nx = 50; p.L = 5 # Space parameters
	p.N = 5; p.kth = 1 # Electronic configuration
	p.Nadd_1 = 4; p.Nadd_N = 2
	p.pod = "den"

	### Max iter
	p.max_iter_inv = 10
	p.max_iter_oda = 30
	p.max_iter_pas = 8
	
	### Pas
	p.pas_init = 0.1 # Initializes
	p.optim_pas = true
	p.learn_pas = true
	p.optimize_pas_ρ_or_G = "ρ"
	
	### Tolerances
	p.tol_lobpcg = 1e-5
	p.tol_oda = 1e-4 
	p.tol_on_ρ = 1e-10

	### Temperature
	p.deg_mult = 8
	p.cool_factor = 1.1
	p.divi = 10

	### Plots
	p.Nx_plots = 50
	p.cutoff_plots = p.Nx

	p
	# then needs to call init_other_params(p) !!
end



function plot_1_and_2(n) # n is 1 or 2
	p = Params()

	### Space and configuration
	p.dim = 1; p.Nx = 100; p.L = 1 # Space parameters
	p.N = 3; p.kth = 1 # Electronic configuration
	p.Nadd_1 = 4; p.Nadd_N = 2
	p.pod = "den"

	### Max iter
	p.max_iter_inv = n==1 ? 200 : 500
	p.max_iter_oda = 30
	p.max_iter_pas = 8
	
	### Pas
	p.pas_init = 0.1 # Initializes
	p.optim_pas = true
	p.learn_pas = true
	p.optimize_pas_ρ_or_G = "ρ"
	
	### Tolerances
	p.tol_lobpcg = 1e-4 
	p.tol_oda = 1e-4 
	p.tol_on_ρ = 1e-10

	### Temperature
	p.deg_mult = 8
	p.cool_factor = 1.1
	p.divi = 10

	init_other_params(p)

	### Produces tρ
	steps = n==1 ? [0,1,2,0.5,4,3,0] : [0,1,0]
	tar = step(steps,p)
	conv = convolve(tar, eval_f(gaussian(p.dim,0.01),p))
	tρ = normalize(abs.(real.(conv)),p.N,p)

	### Inverses
	sol = inverse_pot(tρ,p;plot_infos=true,init_v=-1)

	### Plots for the article
	scale_v = 10
	ts = 2
	shift = n==1 ? 1 : 0.5
	maxy = n==1 ? 10 : 15
	ap = Plots.plot(p.x_axis_cart,[tρ,sol.pot/scale_v .- shift],color=[:blue :red],label=["ρ" string("v/",scale_v)],ylims=(0,maxy),thickness_scaling=ts,legend=:topleft)
	name = n==1 ? "LDA1d" : "LDA1d2"
	# Plots.savefig(ap,string("article_plots/",name,".pdf"))
end

function plot_3_and_4() # n is 1 or 2
	p = Params()

	### Space and configuration
	p.dim = 2; p.Nx = 150; p.L = 5 # L larger enables to have much better convergence, when d=2
	p.N = 2; p.kth = 1
	p.Nadd_1 = 4; p.Nadd_N = 2
	p.pod = "den"

	### Max iter
	p.max_iter_inv = 150
	p.max_iter_oda = 30
	p.max_iter_pas = 3
	
	### Pas
	p.pas_init = 0.1 # Initializes
	p.optim_pas = true
	p.learn_pas = false
	p.optimize_pas_ρ_or_G = "ρ"
	
	### Tolerances
	p.tol_lobpcg = 1e-4
	p.tol_oda = 1e-5
	p.tol_on_ρ = 1e-10

	### Temperature
	p.deg_mult = 3
	p.cool_factor = 1.03
	p.divi = 4

	init_other_params(p)

	### Produces tρ
	steps = [[0.01,0.01,0.01,0.01] [0.01,1,2,0.01] [0.01,3,7,0.01] [0.01,0.01,0.01,0.01]]
	tar = step(steps,p)
	conv = convolve(tar, eval_f(gaussian(p.dim,0.02),p))
	tρ = normalize(abs.(real.(conv)),p.N,p)

	### Inverses
	sol = inverse_pot(tρ,p;plot_infos=true,init_v=0)

	### Plots for the article
	scale_v = 10
	co = [:green,:yellow,:red]
	ts = 1.5
	p_ρ = Plots.heatmap(p.x_axis_cart,p.x_axis_cart,     tρ,grid=false,c=cgrad(co),thickness_scaling=ts,aspect_ratio=:equal,xlims=extrema(p.x_axis_cart),ylims=extrema(p.x_axis_cart),clims=(0,maximum(tρ)))
	p_v = Plots.heatmap(p.x_axis_cart,p.x_axis_cart,sol.pot,grid=false,c=cgrad(co),thickness_scaling=ts,aspect_ratio=:equal,xlims=extrema(p.x_axis_cart),ylims=extrema(p.x_axis_cart))
	# Plots.savefig(p_ρ,"article_plots/LDA2dDenN2k0.png")
	# Plots.savefig(p_v,"article_plots/LDA2dPot.png")
end

function plots_figure_5(k) # n is 1 or 2
	p = Params()

	### Space and configuration
	p.dim = 1; p.Nx = 50; p.L = 5 # Space parameters
	p.N = 5; p.kth = k # Electronic configuration
	p.Nadd_1 = 4; p.Nadd_N = 2
	p.pod = "pot"

	### Max iter
	p.max_iter_inv = 100
	p.max_iter_oda = 30
	p.max_iter_pas = 8
	
	### Pas
	p.pas_init = 0.1 # Initializes
	p.optim_pas = true
	p.learn_pas = true
	p.optimize_pas_ρ_or_G = "ρ"
	
	### Tolerances
	p.tol_lobpcg = 1e-4
	p.tol_oda = 1e-4 
	p.tol_on_ρ = 1e-10

	### Temperature
	p.deg_mult = 8
	p.cool_factor = 1.1
	p.divi = 10

	### Plots
	p.Nx_plots = 500
	p.cutoff_plots = p.Nx

	init_other_params(p)

	### Produces tv and tρ
	σ = 0.1
	vfun(x) = -4*(exp(-(x-0.4)^2/(2*σ^2))) - 2*exp(-(x-0.7)^2/(2*0.1*σ^2)) + 2*exp(-(x-0.5)^2/(2*2*σ^2))
	vf = periodizes_function(vfun,1,p.dim,2) # f becomes a-periodic
	conf_pot = real.(convolve(confining_potential(p,20,5),eval_f(gaussian(p.dim,0.02),p)))
	tv = conf_pot+2*eval_f(vf,p)
	# tv = eval_f(vf,p)
	tv = tv .- minimum(tv)
	(E,ψs) = solve_one_body(tv,p)
	configs = sort_many_body_energies(E,p.N,p.kth,p.Nadd_N)
	tρ = density_of_config(ψs,configs[p.kth].conf,p)
	tρ = normalize(tρ,p.N,p)

	### Inverses
	sol = inverse_pot(tρ,p;plot_infos=true,init_v=-1,tv=tv)

	### Plots for the article
	scale_v = 10
	ts = 1.3
	shift = 0
	maxy = 2.5
	scale_v = 5
	leg_ab = p.kth==1 ? :topright : false

	tρ = abs.(interpolate(tρ,p))
	tv = interpolate(tv,p)
	sol_ρ = abs.(interpolate(sol.ρ,p))
	sol_pot = interpolate(sol.pot,p)
	pot_den = Plots.plot(p.x_axis_cart_plots,[tρ,sol_ρ,sol_ρ,tv/scale_v,sol_pot/scale_v],linestyle=[:dash :solid :solid :dash :solid],color=[:blue :blue :blue :red :red],label=["Target ρ" "Mixed state ρ" "Pure state ρ" LaTeXString(string("Target \$ v/",scale_v,"\$")) LaTeXString(string("\$ v/",scale_v,"\$"))],ylims=(0,maxy),thickness_scaling=ts,legend=leg_ab)
	err_den = log10.(abs.(sqrt.(tρ).-sqrt.(sol_ρ)))
	err_pot = log10.(abs.(tv .- sol_pot))
	leg_err = p.kth==1 ? :topright : false
	errors = Plots.plot(p.x_axis_cart_plots,[err_den,err_pot],color=[:blue :red],label=[LaTeXString("\$ \\log_{10} | \\sqrt{\\rho}-\\sqrt{\\rho_n} \\; |\$") LaTeXString("\$ \\log_{10} | v-v_n \\; |\$")],thickness_scaling=ts,legend=leg_err)
	name_ab = string("article_plots/abs1d5N",p.kth,"k.pdf")
	name_err = string("article_plots/error1d5N",p.kth,"k.pdf")
	# Plots.savefig(pot_den,name_ab)
	# Plots.savefig(errors,name_err)
end

function plot_fig_6()
	p = Params()

	### Space and configuration
	p.dim = 2; p.Nx = 15; p.L = 5
	p.N = 5; p.kth = 1
	p.Nadd_1 = 4; p.Nadd_N = 2
	p.pod = "pot"

	### Max iter
	p.max_iter_inv = 3000
	p.max_iter_oda = 30
	p.max_iter_pas = 5
	
	### Pas
	p.pas_init = 0.1 # Initializes
	p.optim_pas = true
	p.learn_pas = false
	p.optimize_pas_ρ_or_G = "ρ"
	
	### Tolerances
	p.tol_lobpcg = 1e-4
	p.tol_oda = 1e-5
	p.tol_on_ρ = 1e-10

	### Temperature
	p.deg_mult = 3
	p.cool_factor = 1.03
	p.divi = 4

	### Plots
	p.Nx_plots = 200
	p.cutoff_plots = p.Nx

	init_other_params(p)

	### Produces tv and tρ
	σ = 0.05
	vfun(x,y) = -2*(exp(-norm([x,y] .- [0.3,0.7])^2/(2*1.5*σ^2))) - 1.5*(exp(-norm([x,y] .- [0.65,0.35])^2/(2*σ^2)))+ 2*(exp(-norm([x,y] .- [0.75,0.65])^2/(2*1.5*σ^2))) + 2*(exp(-norm([x,y] .- [0.3,0.35])^2/(2*0.8*σ^2))) + 0.3*norm([x,y].-[0.5,0.5])^2
	vf = periodizes_function(vfun,1,p.dim,2) # f becomes a-periodic
	tv = 2*eval_f(vf,p)
	# tv = eval_f(vf,p)
	tv = tv .- minimum(tv)
	(E,ψs) = solve_one_body(tv,p)
	configs = sort_many_body_energies(E,p.N,p.kth,p.Nadd_N)
	tρ = density_of_config(ψs,configs[p.kth].conf,p)
	tρ = normalize(tρ,p.N,p)

	### Inverses
	sol = inverse_pot(tρ,p;plot_infos=true,init_v=0,tv=tv)
	px("Finished with L2 distance on ρ ",norms(sol.ρ.-tρ,p))

	### Plots for the article
	scale_v = 10
	co = [:green,:yellow,:red]
	ts = 1.5

	tρ = abs.(interpolate(tρ,p))
	tv = interpolate(tv,p)
	sol_ρ = abs.(interpolate(sol.ρ,p))
	sol_pot = interpolate(sol.pot,p)
	dρ = log10.(abs.(sqrt.(tρ).-sqrt.(sol_ρ)))
	# p_ρ = Plots.heatmap(p.x_axis_cart_plots,p.x_axis_cart_plots,tρ,grid=false,c=cgrad(co),thickness_scaling=ts,aspect_ratio=:equal,xlims=extrema(p.x_axis_cart_plots),ylims=extrema(p.x_axis_cart_plots),clims=(0,maximum(tρ)))
	p_ρ_err = Plots.heatmap(p.x_axis_cart_plots,p.x_axis_cart_plots,dρ,grid=false,c=cgrad(co),thickness_scaling=ts,aspect_ratio=:equal,xlims=extrema(p.x_axis_cart_plots),ylims=extrema(p.x_axis_cart_plots),clims=(-8,-4))
	# p_v = Plots.heatmap(p.x_axis_cart_plots,p.x_axis_cart_plots,tv,grid=false,c=cgrad(co),thickness_scaling=ts,aspect_ratio=:equal,xlims=extrema(p.x_axis_cart_plots),ylims=extrema(p.x_axis_cart_plots))
	p_v_err = Plots.heatmap(p.x_axis_cart_plots,p.x_axis_cart_plots,log10.(abs.(tv.-sol_pot)),grid=false,c=cgrad(co),thickness_scaling=ts,aspect_ratio=:equal,xlims=extrema(p.x_axis_cart_plots),ylims=extrema(p.x_axis_cart_plots),clims=(-6.5,-1))
	# Plots.savefig(p_ρ,"article_plots/Fig2d_den.png")
	Plots.savefig(p_ρ_err,"article_plots/Fig2d_errDen.png")
	# Plots.savefig(p_v,"article_plots/Fig2d_pot.png")
	Plots.savefig(p_v_err,"article_plots/Fig2d_errPot.png")
end

function figure_TF_pot()
	# subarray where we take one val over k

	function v_ratio(v,p)
		subv = v[Int(floor(p.Nx/3)):Int(floor(2*p.Nx/3))]
		abs(sum(subv)/length(subv))
	end

	#Ns = [60]
	Ns = [2,3,4,5,8,10,15]#,12,15,20,25]#[2,5,10,15,20]
	ns = length(Ns)
	pots_rhos = []
	pars = []
	ratios = []
	ratios_N = []
	first_ratio = 1
	vinv = []

	p = init_p()
	p.Nx = 90
	p.kth = 1
	p.max_iter_inv = 60
	p.pas_init = 50 # Initializes
	p.Nx_plots = 300
	p.Nadd_1 = 3

	# Produces ρ
	σ = 1/5
	ρf0(x) = exp(-(x-0.28)^2/((σ/2)^2)) + 1.3*exp(-(x-0.7)^2/((σ/2)^2)) + 0.5*exp(-(x-0.5)^2/((σ/2)^2))
	ρf = periodizes_function(ρf0,1,p.dim,3)
	p.N = 1; init_other_params(p)
	tρ = normalize(eval_f(ρf,p),1,p)

	for n=1:ns
		p.N = Ns[n]
		init_other_params(p)
		cTF = (4π^2)*(p.dim/(2π))^(2/p.dim)

		if n==1
			vinv = -cTF*tρ.^2
			#vinv = -N*tρ.*log.(tρ) #invert_one_body_density(tρ,p,false)
			vinv .-= minimum(vinv) 
		end

		px("N= ",p.N)
		sol = inverse_pot(p.N*tρ,p;plot_infos=true,init_v=-cTF*(p.N*tρ).^(2/p.dim))

		v = sol.pot
		#val_ratio = abs(v[m])
		val_ratio = v_ratio(v,p)
		if n==1
			first_ratio = abs(val_ratio) #abs(vinv[middle])
			px("ratio vinv ",v_ratio(vinv,p))
			vinv ./= v_ratio(vinv,p)
		end
		push!(ratios,val_ratio/first_ratio)
		push!(ratios_N,p.N/Ns[1])
		push!(pots_rhos,[v/val_ratio,sol.ρ/p.N])
		a = 1
	end

	colors = [:red,:brown,:cyan,:yellow,:green,:purple,:pink]
	tρ_graph = abs.(interpolate(tρ,p))
	plo = Plots.plot(p.x_axis_cart_plots,tρ_graph,label=string(L"\rho"),linewidth = 0.5,size=(1000,500),color=:blue,legend=:topleft,ylims=(0,1))
	scale_v = 2
	px("Nombre plots ",ns)
	subar(a,k) = [a[j] for j=1:k:length(a)]
	for j=1:ns
		l = 1
		vi = real(interpolate(pots_rhos[j][1],p))
		ρi = abs.(interpolate(pots_rhos[j][2],p))
		Plots.plot!(plo,p.x_axis_cart_plots,vi/scale_v,label=latexstring("\$ N=",Ns[j],"\$"),color=colors[j],linewidth = 1)
		# Plots.plot!(plo,p.x_axis_cart_plots,ρi,label=string("N=",Ns[j]),color=colors[j],linewidth = 1)
	end
	Plots.plot!(p.x_axis_cart_plots,real.(interpolate(vinv,p))/scale_v,color=:black,label=L"- c_{TF} \rho^{2/d}/10",linewidth = 0.5)
	px("Ratios ",ratios)
	px("Ratios/N ",ratios./Ns)
	px("N ",ratios_N)
	display([ratios[n]/ratios[end] for n=1:length(ratios)])
	display([(Ns[n]/Ns[end])^(2/p.dim) for n=1:length(Ns)])
	Plots.savefig(plo,"article_plots/plot_TF.pdf")
end

function cusp()
	p = init_p()
	p.Nx = 500
	p.kth = 1
	p.max_iter_inv = 1000
	p.pas_init = 1
	p.Nadd_1 = 3

	init_other_params(p)

	### Produces tρ
	ρf0(x) = exp(-5*abs((x-0.5))) * exp(-10*(x-0.5)^2)
	ρf = periodizes_function(ρf0,1,p.dim,3)
	tρ = normalize(abs.(eval_f(ρf,p)),p.N,p)

	### Inverses
	sol = inverse_pot(tρ,p;plot_infos=true,init_v=0)

	### Plots for the article
	scale_v = 20
	ts = 2
	shift = 0
	maxy = 10
	size = (700,500)
	v = sol.pot/scale_v .- shift#real(interpolate(sol.pot,p))/scale_v .- shift
	ρ_pl = tρ #real(interpolate(tρ,p))
	ap = Plots.plot(p.x_axis_cart,[ρ_pl,v],color=[:blue :red],label=["ρ" string("v/",scale_v)],thickness_scaling=ts,legend=:topleft,size=size)#,ylims=(0,maxy))
	name = "cusp"
	Plots.savefig(ap,string("article_plots/",name,".pdf"))

	err_den = log10.(abs.(sqrt.(tρ).-sqrt.(sol.ρ)))
	errors = Plots.plot(p.x_axis_cart,err_den,color=:blue,label=LaTeXString("\$ \\log_{10} | \\sqrt{\\rho}-\\sqrt{\\rho_n} \\; |\$"),thickness_scaling=ts,size=size)
	Plots.savefig(errors,string("article_plots/error_",name,".pdf"))
end


function other(n) # n is 1 or 2
	# Main parameters
	dim = 1
	Nx = 300 # discretization number
	N = 2 # number of electrons
	kth = 1 # number of the state we look (1 for ground)
	L = 1 # physical length of cube
	Nadd_1 = 4 # number of additional 1-body states we will compute, to be sure we get the degeneracies
	Nadd_N = 2 # additional 1-body orbitals for computing many-body energies, should be computed automatically
	pas_init = 0.1
	pod = "den"
	temps_ratio = 8
	cool_factor = 1.1
	max_niter_oda = 30
	max_niter_particle_swarm = 30
	optimiser_pas = true
	max_iter_pas = 8
	learn_pas = true
	optimize_pas_ρ_or_G = "ρ"
	divi = 10
	tol = 1e-4 # for LOBPCG (one body) and ODA
	tol_on_ρ = 1e-5 # error in the density with respect to the target density

	p = Params(dim,Nx,tol,N,kth,Nadd_1,Nadd_N,temps_ratio,-1,L,pod,pas_init,learn_pas,optimiser_pas)
	p.max_iter_inv = 100

	vf(x) = 0 # function in [0,1]^d
	if dim == 1
		σ = 1/8
		vf(x) = -3*(exp(-(x-0.5)^2/(2*σ^2))) - 2*(exp(-(x-0.3)^2/(2*0.2*σ^2)))
	elseif dim == 2
		vf(x,y) = exp(-(x^2+y^2)/0.02)
		#vf(x,y) = 3*exp(-((x-2/7)^2+(y-1/2)^2)/0.01)+exp(-((x-1/2)^2+(y-1/2)^2)/0.02)+3*exp(-((x-5/7)^2+(y-1/2)^2)/0.02)+7*exp(-((x-6/7)^2+(y-1/2)^2)/0.005)
	elseif dim == 3
		vf(x,y,z) = 
		exp(-((x-0.2/2)^2+(y-1.1/2)^2+(z-0.8/2)^2)/0.02) 
		+ exp(-((x-1.2/2)^2+(y-0.8/2)^2+(z-0.9/2)^2)/0.01) 
		+ exp(-((x-0.8/2)^2+(y-1.2/2)^2+(z-1.2/2)^2)/0.005)
	end
	vfun = periodizes_function(vf,1,dim,3) # f becomes a-periodic

	tv = zeroar(p,false)

	if p.pod=="pot"
		#conf_pot = convolve(confining_potential(p,60,300),eval_f(gaussian(p.dim,0.02),p),p.dx)
		#tv = conf_pot-6*eval_f(vfun,p) + conf_pot
		tv = eval_f(vfun,p)
		tv = tv .- minimum(tv)
		(E,psis) = solve_one_body(tv,p)
		configs = sort_many_body_energies(E,p.N,p.kth,p.Nadd_N)
		tρ = density_of_config(psis,configs[p.kth].conf,p)
		tρ = normalize(tρ,p.N,p)

	elseif p.pod=="den"
		#tρ = normalize(eval_f(zero_borders(vfun,100),p),p.N,p)
		tar = eval_f(vfun,p)
		tρ = normalize(tar,p.N,p)
		#tρ = normalize(convolve(dirichlet(eval_f(vfun,p),1/30), eval_f(gaussian(p.dim,0.01),p),p.dx),p.N,p)
		# Step function
		if true
			if p.dim==1
				steps = [0.01,1,2,0.5,0.01]
			elseif p.dim==2
				steps = [[0.01,0.01,0.01,0.01] [0.01,1,2,0.01] [0.01,3,7,0.01] [0.01,0.01,0.01,0.01]]
			end
			tar = step(steps,p)
			conv = convolve(tar, eval_f(gaussian(p.dim,0.01),p))
			tρ = normalize(abs.(real.(conv)),p.N,p)
		end
	end

	inverse_pot(tρ,tol_on_ρ,p;plot_infos=true,init_v=0)

end

# plot_1_and_2(1)
# plot_1_and_2(2)
# plot_3_and_4()
# plots_figure_5(2)
# plot_fig_6()
# figure_TF_pot()
cusp()
a = 1
