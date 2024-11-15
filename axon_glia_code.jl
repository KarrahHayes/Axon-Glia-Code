using GLMakie
using GLM
using DataFrames
using Statistics

function add_intercept_column(x::AbstractVector{T}) where {T}
    mat = similar(x, float(T), (length(x), 2))
    fill!(view(mat, :, 1), 1)
    copyto!(view(mat, :, 2), x)
    return mat
end
function fit_line!(ax, x, y)
    df = DataFrame(x=x, y=y)
    modd = lm(@formula(y ~ x), df)
    pred = GLM.predict(modd, add_intercept_column(df.x); interval=:confidence, level=0.95)
    inds = sortperm(df.x)
    band!(ax, df.x[inds], pred.lower[inds], pred.upper[inds];
        color=(:gray, 0.5),
        alpha=0.5)
    lines!(ax, df.x, pred.prediction, color=:gray, linewidth=7)
end

function read_axon_file(filepath)
    coordlines = readlines(filepath)
    start_lines = findall(occursin.("\t\tx\ty\tz", coordlines))
    end_lines = findall(occursin.("\t\tClosed/Open length", coordlines))
    axonxcoord, axonycoord = [], []
    for i in eachindex(start_lines)
        coordinates = split.(coordlines[start_lines[i]:end_lines[i]], "\t")
        notextcoordinates = coordinates[2:end-1]
        thisconx = Float64[]
        thiscony = Float64[]
        for line in notextcoordinates
            push!(thisconx, parse.(Float64, line[3]))
            push!(thiscony, parse.(Float64, line[4]))
        end  
        push!(axonxcoord, thisconx)
        push!(axonycoord, thiscony)
    end
    return axonxcoord, axonycoord
end
xcoord1, ycoord1 = read_axon_file("/Users/karrahhayes/Desktop/Research/pt.1_large_axons.txt")
xcoord2, ycoord2 = read_axon_file("/Users/karrahhayes/Desktop/Research/pt.2_large_axons.txt")
#for coord1
axonarea_pixels1 = []
for i in 1:length(xcoord1)
    A = 0.5 * abs(sum((ycoord1[i][j] + ycoord1[i][j+1]) * (xcoord1[i][j] - xcoord1[i][j+1]) for j in 1:length(xcoord1[i])-1))
    push!(axonarea_pixels1, A)
end
axonarea_nm1 = 13.4441 * 13.4441 .* axonarea_pixels1
axonarea_um1 = axonarea_nm1./1e6
#for coord2
axonarea_pixels2 = []
for i in 1:length(xcoord2)
    A = 0.5 * abs(sum((ycoord2[i][j] + ycoord2[i][j+1]) * (xcoord2[i][j] - xcoord2[i][j+1]) for j in 1:length(xcoord2[i])-1))
    push!(axonarea_pixels2, A)
end
axonarea_nm2 = 24.8963 * 24.8963 .* axonarea_pixels2
axonarea_um2 = axonarea_nm2./1e6
allaxonarea_um = append!(axonarea_um1, axonarea_um2)
allaxonradii_um = sqrt.(largeaxonarea_um./pi)
allaxondiameter_um = (2 .* allaxonradii_um)
# large axons classified as having diameter above 10 um (or like very close to 10)
largeaxondiameter_um = allaxondiameter_um[allaxondiameter_um .> 9.5]
#=
#populating coordinates of axons and glia into dictionary
coordlines = readlines("/Users/karrahhayes/Desktop/Research/sample_info.txt")
objectindices = findall(occursin.("OBJECT", coordlines))
objectnames = coordlines[findall(occursin.("OBJECT", coordlines)) .+ 1] 
output_dict = Dict()
for i = 1:length(objectnames)
    for (j, object) in enumerate(objectnames)
        objxcoord, objycoord = [], []
        startindex = objectindices[i] 
        if i == length(objectindices)
            endindex = length(coordlines)
        else
            endindex = objectindices[i+1]
        end
        subcoordinates = coordlines[startindex:endindex]
        start_lines = findall("\t\tx\ty\tz" .== subcoordinates)
        end_lines = findall(occursin.("\t\tClosed/Open length", subcoordinates))
        # Assumes start_lines and end_lines find the same number of elements!
        for i in eachindex(start_lines)
            coordinates = split.(subcoordinates[(start_lines[i]+1):(end_lines[i]-1)], "\t")
            thisx = Float64[]
            thisy = Float64[]
            for line in coordinates
                not_empty = findall((!).(isempty.(line)))
                push!(thisx, parse.(Float64, line[not_empty[1]]))
                push!(thisy, parse.(Float64, line[not_empty[2]]))
            end
        push!(objxcoord, thisx)
        push!(objycoord, thisy)
        end
    end
    push!(output_dict, (objectnames[i]) => [objxcoord, objycoord])
end
axonxcoord = output_dict["NAME:  whole axons"][1]
axonycoord = output_dict["NAME:  whole axons"][2]
gliaxcoord = output_dict["NAME:  glia"][1]
gliaycoord = output_dict["NAME:  glia"][2]

#computing area using xy coordinates of contours
#area of axons
axonarea_pixels = []
for i in 1:length(axonxcoord)
    A = 0.5 * abs(sum((axonycoord[i][j] + axonycoord[i][j+1]) * (axonxcoord[i][j] - axonxcoord[i][j+1]) for j in 1:length(axonxcoord[i])-1))
    push!(axonarea_pixels, A)
end
axoncircumference_pixels = []
for i in 1:length(axonxcoord)
    C = sum(sqrt.(diff(axonxcoord[i]).^2 .+ diff(axonycoord[i]).^2))
    push!(axoncircumference_pixels, C)
end
axonarea_nm = 5.102 * 5.102 .* axonarea_pixels
axonarea_um = axonarea_nm./1e6
axonradius_um = sqrt.(axonarea_um./pi)
axonradius_m = axonradius_um./1e6
axonradius_pixels = sqrt.(axonarea_pixels./pi)
axonCOM_x = mean.(axonxcoord)
axonCOM_y = mean.(axonycoord)
axoncircumference_nm = 5.102 .* axoncircumference_pixels
axoncircumference_um = axoncircumference_nm./1000
axonroundness = (4 * pi .* axonarea_um)./(axoncircumference_um .^2)
#area of glia and axons
gliaarea_pixels = []
for i in 1:length(gliaxcoord)
    A = 0.5 * abs(sum((gliaycoord[i][j] + gliaycoord[i][j+1]) * (gliaxcoord[i][j] - gliaxcoord[i][j+1]) for j in 1:length(gliaxcoord[i])-1))
    push!(gliaarea_pixels, A)
end
gliaarea_nm = 5.102 * 5.102 .* gliaarea_pixels
gliaarea_um = gliaarea_nm./1e6
gliaradius_um = sqrt.(gliaarea_um./pi)
gliaradius_pixels = sqrt.(gliaarea_pixels./pi)
gliaCOM_x = mean.(gliaxcoord)
gliaCOM_y = mean.(gliaycoord)

#distribution of axon radii
update_theme!(fontsize = 35)
r = Figure()
minval,  maxval = extrema(axonradius_um)
axr = Axis(r[1,1], xlabel = "Axon Radius (um)", title = "Axon Size", ylabel = "Probability", xscale = log10)
hist!(axr, axonradius_um, color = RGBf(0.196, 0.553, 0.753), bins = exp.(LinRange(log(minval), log(maxval), 80)), normalization = :probability)
r 

n = Figure()
minval,  maxval = extrema(axonarea_um)
axn = Axis(n[1,1], xlabel = "Axon Area (um^2)", title = "Axon Area", ylabel = "Probability", xscale = log10)
hist!(axn, axonarea_um, color = :blue, bins = exp.(LinRange(log(minval), log(maxval), 80)), normalization = :probability)
#Cm must be entered in F/m^2, Rm in ohm-m^2, and Ra in ohm-m
cv(Cm, Rm, Ra, axonradius_m) = (sqrt.((Rm .* axonradius_m)./(2 .* Ra)))./(Cm .* Rm)
axcv = Axis(n[1,2], xlabel = "Axon Area (m^2)", ylabel = "Estimated Conduction Velocity (m/sec)")
vx = axonarea_nm./1e18 #in m
vy = cv(0.01, 0.21, 1, axonradius_m)
scatter!(axcv, vx, vy, color = :green)
n

#axon circularity
c = Figure()
axc = Axis(c[1,1], xlabel = "Log of Axon Radius (um)", ylabel = "Axon Circularity", title = "Axon Size vs. Axon Circularity")
ylims!(axc, (0,1))
scatter!(axc, log10.(axonradius_um), axonroundness, color = :red)
c

#mitochondria count
mitolines = readlines("/Users/karrahhayes/Desktop/Research/mitochondria_sample.txt")
splitlines = split.(mitolines[contains.(mitolines, "CONTOUR")], "0  ")
mitocount = zeros(length(splitlines))
for (i,line) in enumerate(splitlines)
    mitocount[i] = parse(Float64,split(line[2], " points")[1])
end
mitox_ind = []
#= for i = 1:length(axoncircumference_um)
    if (axoncircumference_um[i] ./ (2*pi)) > 0.5
        push!(mitox_ind, i)
    end
end =#
for i = 1:length(mitocount)
    if mitocount[i] >= 2
        push!(mitox_ind, i)
    end
end
mx = log10.(axoncircumference_um[mitox_ind])
my = log10.(mitocount[mitox_ind])
findinf = isfinite.(my)
purged_my = my[findinf]
purged_mx = mx[findinf]
m = Figure()
axm = Axis(m[1,1], xlabel = "Log of Axon Circumference (um)", ylabel = "Log of Number of Mitochondria", title = "Axon Size vs. Mitochondria")
scatter!(axm, purged_mx, purged_my, color = :orange)
fit_line!(axm, purged_mx, purged_my)
m
#idk why but cant get coeffs when within function so this is to access values 
df = DataFrame(purged_mx=purged_mx, purged_my=purged_my)
modd = lm(@formula(purged_my ~ purged_mx), df)
pred = GLM.predict(modd, add_intercept_column(df.purged_mx); interval=:confidence, level=0.95)
inds = sortperm(df.purged_mx)
band!(axm, df.purged_mx[inds], pred.lower[inds], pred.upper[inds];
    color=(:gray, 0.5),
    alpha=0.5)
lines!(axm, df.purged_mx, pred.prediction, color=:gray, linewidth=7)
mitocoeffs = DataFrame(coeftable(modd))
mito_correl_coeff = cor(purged_mx, purged_my)
mito_slope = mitocoeffs[2, 2]
mitoslope_CI = mito_slope - mitocoeffs[2,6]
mito_yint = mitocoeffs[1, 2]

#matching glia to its axon
dist = zeros(length(gliaCOM_x))
mask = zeros(length(gliaCOM_x))
matchingaxon = []
for i in eachindex(gliaCOM_x)
    thisradius = gliaradius_pixels[i]
    dist = sqrt.((gliaCOM_x[i] .- axonCOM_x).^2 + (gliaCOM_y[i] .- axonCOM_y).^2)
    index = findall(dist .< thisradius)
    push!(matchingaxon, index)
end
popfirst!(matchingaxon[1])
pop!(matchingaxon[1])
matchingaxon[47] = 277
matchingaxon[97] = 37
#these lines use hardcoding to remove issues in matchingaxon
#will likely need to readdress with new file
matchedaxon = collect(Iterators.flatten(matchingaxon))

# calculating thickness
gliathickness_um = []
assoc_axon_radius = []
for (i, line) in enumerate(matchedaxon)
    gliathickness_um = abs.(gliaradius_um .- axonradius_um[line])
    push!(assoc_axon_radius, axonradius_um[line])
end

# g-ratio plot
gratio = (assoc_axon_radius .* 2)./(gliaradius_um .* 2)
gr = Figure()
axgr = Axis(gr[1,1], xlabel = "Fiber Diameter (um)", ylabel = "g-Ratio", title = "Axon Radius vs. g-Ratio")
ylims!(axgr, (0,1))
scatter!(axgr, gliaradius_um .* 2, gratio, color = :magenta)
gr

#correlation between axon area and glia thickness
ggx = log10.(assoc_axon_radius)
g_ind = []
for i = 1:length(ggx)
    if ggx[i] > 0
        push!(g_ind, i)
    end
end
gx = ggx[g_ind]
gy = log10.(gliathickness_um)[g_ind]
ggy = log10.(gliathickness_um)
g = Figure()
axg = Axis(g[1,1], xlabel = "Log of Axon Radius (um)", ylabel = "Log of Glia Thickness (um)", title = "Relationship Between Axon Size and Surrounding Glia Thickness")
scatter!(axg, ggx, ggy, color = :green)
#=fit_line!(axg, ggx, ggy)
g 
#idk why but cant get coeffs when within function so this is to access values 
df = DataFrame(gx=gx, gy=gy)
modd = lm(@formula(gy ~ gx), df)
pred = GLM.predict(modd, add_intercept_column(df.gx); interval=:confidence, level=0.95)
inds = sortperm(df.gx)
band!(axg, df.gx[inds], pred.lower[inds], pred.upper[inds];
    color=(:gray, 0.5),
    alpha=0.5)
lines!(axg, df.gx, pred.prediction, color=:gray, linewidth=7)
gliacoeffs = DataFrame(coeftable(modd))
glia_correl_coeff = cor(gx, gy)
glia_slope = gliacoeffs[2, 2]
gliaslope_CI = glia_slope - gliacoeffs[2, 6]
glia_yint = gliacoeffs[1, 2] =#

#distribution of axon positions
p = Figure()
xmin, xmax = extrema(axonCOM_x)
ymin, ymax = extrema(axonCOM_y)
axp = Axis(p[1,1], title = "Axon Positions", xlabel = "X-Coordinate (px)", ylabel = "Y-Coordinate (px)")
scatter!(axp, axonCOM_x, axonCOM_y, markersize = axonradius_pixels.*2, markerspace = :data, color = axonradius_pixels)
p

#estimating all possible values of rm and ra
t = Figure()
x = range(1, 1e9, 10)
axt = Axis(t[1,1], xlabel = "ra (Ω/m)", ylabel = "rm (Ω-m)")
vspan!(t[1,1], [1.149e9], [9.94e9])
hspan!(t[1,1], [2109], [11248])
for cv in range(0.1, 10, 6)
    ablines!(axt, 0, ((cv * 0.0017).^2))
end
xlims!(axt, (0, nothing))
t
=#