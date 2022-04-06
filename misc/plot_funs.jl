using Plots#, PyPlot, GR; pyplot()

function save_plot(ars,p,name)
	pl = Plots.plot()
	n = length(ars)
	if p.dim==1
		# pl_ax = translation(p.x_axis_cart,Int(floor(p.Nx/2)))
		# pl_a = translation(ars,Int(floor(p.Nx/2)))
		pl = Plots.plot(p.x_axis_cart,ars)
	elseif p.dim==2
		hmaps = []
		for i=1:n
			h = Plots.heatmap(p.x_axis_cart,p.x_axis_cart,ars[i],size=(500,700))
			push!(hmaps,h)
		end
		pl = Plots.plot(hmaps..., layout = (n,1))
	elseif p.dim==3
		pl = Plots.plot(ars[1])
	end
	Plots.savefig(pl,string(name,".png"))
end

plAr(ar,p) = display(plot(p.x_axis_cart, ar))

function plPotStates(pot,eigenvals,eigenfuns,p) # prints pot and eigenvalues
	plo = plot(p.x_axis_cart, pot/maximum(pot), label="pot")
	for i = 1:size(eigenfuns,1)
		plot!(plo,p.x_axis_cart,eigenfuns[i], label=string(i," ",eigenvals[i]))
	end
	display(plo)
end

function plOneFun(f,others,p) # prints one function and a list
	plo = Plots.plot(p.x_axis_cart, f)
	for i = 1:size(others,1)
		plot!(plo,p.x_axis_cart,others[i])
	end
	display(plo)
end

function plArs(ars,p,title="",dashed=[],colors=[],labels=[],mini=0,maxi=0) # prints arrays
	dashed_list = 0
	if dashed == []
		dashed_list = [false for i=1:length(ars)]
	else
		dashed_list = dashed
	end
	plo = Plots.plot(title=title,grid=false)
	if labels == []
		plot!(plo,legend=false)
	else 
		plot!(plo,legend=:topleft,legendfontsize=10)
	end
	middle = floor(Int,p.Nx)
	global_min = 0; global_max = 0
	for i = 1:size(ars,1)
		tp = ars[i]
		colo = colors==[] ? :black : colors[i]
		lab = labels==[] ? "" : labels[i]
		if p.dim==2
			tp = ars[i][:,middle]
		elseif p.dim==3
			tp = ars[i][:,middle,middle]
		end
		global_min = minimum(map(a->minimum(a),tp))
		global_max = maximum(map(a->maximum(a),tp))
		if dashed_list[i]
			plot!(plo,p.x_axis_cart,tp,line=(:dash),color= colo,label=lab)
		else
			plot!(plo,p.x_axis_cart,tp,color= colo,label=lab)
		end
	end
	lims = (mini == 0 && maxi == 0) ? (global_min,global_max) : (mini,maxi)
	plot!(plo,ylims=lims)
	#plot!(plo,ylims=lims,ticks=false) # remove the graduations of the axis
	plo
end

function pl2axis(a,b,p) # prints arrays
	plo = plot(legend=false,grid=false)
	plot!(plo,p.x_axis_cart,a)
	plot!(plo,p.x_axis_cart,b)
	plo
end

function decrease_resolution(a,n) # decreases resolution to be able to plot, decreases the precision of floats, for tikz output
	len = Int(floor(size(a,1)/n))
	na = zeros(len,len)
	for i= 1:len
		for j= 1:len
			na[i,j] = round(a[Int(floor(i*n)),Int(floor(j*n))], digits=1)
		end
	end
	na
end

function plot2d(a,colors=[],title="")
	#na = decrease_resolution(a,1.8)
	co = colors == [] ? [:green,:yellow, :red] : colors
	h = Plots.heatmap(a,c=cgrad(co),axis=([], false),ticks=nothing,grid=false,fontsize=100) #,extra_kwargs =:subplot) #,colormap_name = "viridis") to export in tikz
	return h
end

function save3d(a,p,title="plot3d",title_data="array")
	prefix = "plots/"
	vtk_grid(prefix*title, p.x_axis_cart, p.x_axis_cart, p.x_axis_cart) do vtk
	    vtk[title_data] = a
	end
end

#=
function f()
	#n = size(a,1)
	#f = arToF(a,p) 
	#granu = p.a/20
	#x, y = 0:granu:p.a, 0:granu:p.a
	#z = Surface((x,y)->1,x,y)
	#plo = plot(x,y,z, st = [:contourf],axis=([], false),xaxis=false,yaxis=false,ticks=nothing) #,title="function "*fg*", n="*string(n)
	#savefig(plo,"plots/2d.tikz")
	#plo
end
=#

