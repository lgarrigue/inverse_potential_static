include("derivations.jl")
using Printf, LinearAlgebra
px = println
import Base.+  
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)  

float2str(x,n=3) = @sprintf("%.2E", x) # n is the number of digits
ro(x,n=3) = round(x, sigdigits=n) # n is the number of digits
pr(s) = println("========  ",s)
dims(a) = (ndims(a),size(a,1))

function pxdims(a)
	d = ndims(a)
	px("Dim ",d)
	for i=1:d
		px("Size ",size(a,i)," ")
	end
end

function id_mat(n,complex=false)
	M = zeros(n,n)
	if complex
		M = zeros(ComplexF64,n,n)
		#M = Matrix((1.0+im)I, n, n) 
#px(M)
	else
		M = zeros(n,n)
	end
	for i=1:n
		M[i,i] = 1
	end
	M
end

convolve(f,g) = ifft(fft(f).*fft(g))

function gaussian(d,s,mean=[]) # gaussian
	m = mean==[] ? zeros(d) : mean
	c = (2*π*s)^(-d/2)
	a = 1/(2*s^2)
	f = 0
	if d==1
		f = x -> c*exp(-a*(x-m[1])^2)
	elseif d==2
		f = (x,y) -> c*exp(- a*((x-m[1])^2+(y-m[2])^2))
	elseif d==3
		f = (x,y,z) -> c*exp(- a*((x-m[1])^2+(y-m[2])^2+(z-m[3])^2))
	end
	periodizes_function(f,1,d,3) 
end

x_to_n(x,p) = mod1(1+floor(Int,x/p.dx),p.Nx)

function mean_middle(v,p,n) # mean of the middle, to etalonner potentials, X
	fi = x_to_n(p.L/n,p)
	en = x_to_n(p.L*(1-1/n),p)
	if p.dim == 1
		na = v[fi:en]
	elseif p.dim==2
		na = v[fi:en,fi:en]
	elseif p.dim==3
		na = v[fi:en,fi:en,fi:en]
	end
	sum(na)/length(na)
end

function mean_middleXT(v,p,n) # mean of the middle, to etalonner potentials, X
	x = 0
	for i=1:p.Nt
		x += mean_middle(v[:,t],p,n)
	end
	x/p.Nt
end

function ar_log(m,M,n) # array from m to M (comprised) in log scale, with length n
	α = (M/m)^(1/(n-1))
	[m*α^(i-1) for i=1:n]
end

function translation_function(f,τ,d)
	if d==1
		return x -> f(x-τ[1])
	elseif d==2
		return (x,y) -> f(x-τ[1],y-τ[2])
	elseif d==3
		return (x,y,z) -> f(x-τ[1],y-τ[2],z-τ[3])
	end
end

function periodizes_function(f,a,d,n=2) # f becomes a-periodic
	if d==1
		g = x -> 0
		for i=-n:n
			g = g + translation_function(f,a*[i],d)
		end
		return g
	elseif d==2
		g = (x,y) -> 0
		for i=-n:n
			for j=-n:n
				g = g + translation_function(f,a*[i,j],d)
			end
		end
		return g
	elseif d==3
		g = (x,y,z) -> 0
		for i=-n:n
			for j=-n:n
				g = g + translation_function(f,a*[i,j,l],d)
			end
		end
		return g
	end
end

function translation(a,j)
	d = ndims(a); n = size(a,1)
	b = similar(a)
	if d==1
		for x=1:n
			b[x] = a[mod1(x-j[1],n)]
		end
		return b
	elseif d==2
		for x=1:n
			for y=1:n
				b[x,y] = a[mod1(x-j[1],n),mod1(y-j[2],n)]
			end
		end
		return b
	else
		for x=1:n
			for y=1:n
				for z=1:n
					b[x,y,z] = a[mod1(x-j[1],n),mod1(y-j[2],n),mod1(z-j[3],n)]
				end
			end
		end
		return b
	end
end

# f is a function on [0,1]^d, creates an array on the grid [0,L]^d
function eval_f_raw(f,dim,N,complex=false) 
	x_axis_red = (0:N-1)/N
	if dim == 1
		a = f.(x_axis_red)
		return complex ? a : real.(a)
	else
		a = zeros(complex ? ComplexF64 : Float64,Tuple(fill(N,dim)))
		if dim == 2
			for x=1:N
				for y=1:N
					X = x_axis_red[x]; Y = x_axis_red[y] 
					m = f(X,Y)
					a[x,y] = complex ? f(X,Y) : real(f(X,Y))
				end
			end
		elseif dim == 3
			for x=1:N
				for y=1:N
					for z=1:N
						X = x_axis_red[x]; Y = x_axis_red[y]; Z = x_axis_red[z]
						a[x,y,z] = complex ? f(X,Y,Z) : real(f(X,Y,Z))
					end
				end
			end
		end
	end
	a
end

function fft_to_fun_raw(a,dim,N,k_axis,cutoff)
	if dim == 1
		g(x) = 0
		for k=1:N
			K = k_axis[k]
			if abs(K) ≤ cutoff
				f(x) = a[k]*cis(2π*K*x)
				g = g + f
			end
		end
		return g
	elseif dim == 2
		Gg(x,y) = 0
		for kx=1:N
			Kx = k_axis[kx]
			if abs(Kx) ≤ cutoff
				for ky=1:N
					Ky = k_axis[ky]
					if abs(Ky) ≤ cutoff
						f(x,y) = a[kx,ky]*cis(2π*(Kx*x+Ky*y))
						Gg = Gg + f
					end
				end
			end
		end
		return Gg
	elseif dim == 3
		Gg(x,y,z) = 0
		for kx=1:N
			Kx = k_axis[kx]
			if abs(Kx) ≤ cutoff
				for ky=1:N
					Ky = k_axis[ky]
					if abs(Ky) ≤ cutoff
						for kz=1:N
							Kz = k_axis[kz]
							if abs(Kz) ≤ cutoff
								f(x,y,z) = a[kx,ky,kz]*cis(2π*(Kx*x+Ky*y+Kz*z))
								Gg = Gg + f
							end
						end
					end
				end
			end
		end
		return Gg
	end
end
