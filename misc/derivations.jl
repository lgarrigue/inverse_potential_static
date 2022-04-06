# === Laplacian of a function (array)

function pre_lap(f) # array function
	d = ndims(f)
	n = size(f,1)
	Δf = zeros(Tuple(fill(n,d)))
	if d == 1
		for i=2:n-1
			Δf[i] = f[i+1]+f[i-1]-2*f[i]
		end
	elseif d == 2
		for i=2:n-1
			for j=2:n-1
				Δf[i,j] = f[i+1,j]+f[i-1,j]+f[i,j-1]+f[i,j+1]-4*f[i,j]
			end
		end
	elseif d == 3
		for i=2:n-1
			for j=2:n-1
				for y=2:n-1
					Δf[i,j,y] = f[i+1,j,y]+f[i-1,j,y]+f[i,j-1,y]+f[i,j+1,y]+f[i,j,y-1]+f[i,j,y+1]-6*f[i,j,y]
				end
			end
		end
	end
	Δf
end

function regularize_borders_array(f) # regularization at borders (becomes continuous)
	na = copy(f)
	d = ndims(f)
	n = size(f,1)
	if d == 1
		na[1] = na[2]
		na[n] = na[n-1]
	elseif d == 2
		for i=2:n-1
			na[1,i] = na[2,i]
			na[n,i] = na[n-1,i]
			na[i,1] = na[i,2]
			na[i,n] = na[i,n-1]
		end
		na[1,1] = na[2,2]
		na[1,n] = na[2,n-1]
		na[n,1] = na[n-1,2]
		na[n,n] = na[n,n]
	elseif d == 3
		for i=2:n-1
			for j=2:n-1
				na[1,i,j] = na[2,i,j]
				na[n,i,j] = na[n-1,i,j]
				na[i,1,j] = na[i,2,j]
				na[i,n,j] = na[i,n-1,j]
				na[i,j,1] = na[i,j,2]
				na[i,j,n] = na[i,j,n-1]
			end
		end
	# should do the same for aretes and point extremities
	end
	na
end

function ghost(f) # creates array with size size(f)+2, ghost cells at boundaries
	d = ndims(f)
	n = size(f,1)
	ghost = zeros(Tuple(fill(n+2,d)))
	if d == 1
		for i=2:n+1
			ghost[i] = f[i-1]
		end
	elseif d == 2
		for i=2:n+1
			for j=2:n+1
				ghost[i,j] = f[i-1,j-1]
			end
		end
		for i=2:n+1
			ghost[1,i] = ghost[2,i]
			ghost[n+2,i] = ghost[n+1,i]
			ghost[i,1] = ghost[i,2]
			ghost[i,n+2] = ghost[i,n+1]
		end
	elseif d == 3
		for i=2:n+1
			for j=2:n+1
				for l=2:n+1
					ghost[i,j,l] = f[i-1,j-1,l-1]
				end
			end
		end
	end
	regularize_borders_array(ghost)
end




function Δ(f,dx)
	d = ndims(f)
	n = size(f,1)
	g = ghost(f)
	pre = pre_lap(g)
	c = (1/dx)^2
	if d==1
		return c*pre[2:n+1]
	elseif d==2
		return c*pre[2:n+1,2:n+1]
	elseif d==3
		return c*pre[2:n+1,2:n+1,2:n+1]
	end
end


function ΔmatPer(n,dx) # matrix -2 1 0 ... 0 1 ; ... ; 1 0 ... 1 -2 1 ... for periodic boundary conditions, it's -Δ
	Δ = zeros(n,n)
	for i=1:n
		for j=1:n
			Δ[i,j] = i==j ? -2 : (abs(i-j) == 1 ? 1 : 0)
		end
	end
	Δ[1,end] = 1
	Δ[end,1] = 1
	Δ/(dx^2) 
end


function ∇matPerSym(n,dx)
	∇ = zeros(n,n)
	for i=1:n
		for j=1:n
			∇[i,j] = abs(i-j)==1 ? j-i : 0
		end
	end
	∇[1,end] = -1
	∇[end,1] = 1
	∇/(2*dx)
end

function ∇matPer(n,dx)
	∇ = zeros(n,n)
	for i=1:n
		for j=1:n
			∇[i,j] = j-i==1 ? 1 : j==i ? -1 : 0
		end
	end
	∇[end,1] = 1
	∇/dx
end

function Δper(f,dx,complex=false) # first point is equal to last + 1 point
	d = ndims(f); n = size(f,1)
	Δf = zeros(complex ? ComplexF64 : Float64, Tuple(fill(n,d)))
	if d == 1
		for i=1:n
			Δf[i] = f[mod1(i+1,n)]+f[mod1(i-1,n)]-2*f[i]
		end
	elseif d == 2
		for i=1:n
			for j=1:n
				Δf[i,j] = f[mod1(i+1,n),j]+f[mod1(i-1,n),j]+f[i,mod1(j-1,n)]+f[i,mod1(j+1,n)]-4*f[i,j]
			end
		end
	elseif d == 3
		for i=1:n
			for j=1:n
				for l=1:n
					Δf[i,j,l] = f[mod1(i+1,n),j,l]+f[mod1(i-1,n),j,l]+f[i,mod1(j-1,n),l]
					+f[i,mod1(j+1,n),l]+f[i,j,mod1(l-1,n)]+f[i,j,mod1(l+1,n)]-6*f[i,j,l]
				end
			end
		end
	end
	Δf/(dx^d)
end

function ∇(a,dx)
	gr = []
	n = size(a,1)
	for i=1:n-1
		x = a[i+1]-a[i]
		push!(gr,x)
	end
	push!(gr,gr[end])
	gr/dx
end

#= TEST RIGHT LAPLACIAN OK
den(x) = exp(-(x-a/2)^2)
array_den = fToAr(den,P,a)
del_den(x) = (exp(-(x-a/2)^2))*(4*(x-a/2)^2-2)
array_del_den = fToAr(del_den,P,a)

plot(array_del_den,label="theoretical")
del = Δ(array_den,P.dx)
display(plot!(del,label="computed"))
=#
