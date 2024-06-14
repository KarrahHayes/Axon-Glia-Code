using GLMakie
using GLM
using DataFrames

##

function read_contours(filename)
    coordlines = readlines(filename)
    axonxcoord, axonycoord = [], []
    start_lines = findall("\t\tx\ty\tz" .== coordlines)
    end_lines = findall(occursin.("\t\tClosed/Open length", coordlines))
    # Assumes start_lines and end_lines find the same number of elements!
    for i in eachindex(start_lines)
        coordinates = split.(coordlines[(start_lines[i]+1):(end_lines[i]-1)], "\t")
        thisx = Float64[]
        thisy = Float64[]
        for line in coordinates
            not_empty = findall((!).(isempty.(line)))
            push!(thisx, parse.(Float64, line[not_empty[1]]))
            push!(thisy, parse.(Float64, line[not_empty[2]]))
        end
    push!(axonxcoord, thisx)
    push!(axonycoord, thisy)
    end
    return axonxcoord, axonycoord
end

#old method of computing area
axonlines = readlines("/Users/karrahhayes/Desktop/Research/axon size.txt")
split_lines = split.(axonlines[contains.(axonlines, "CONTOUR")], ",")
axonlengths = zeros(length(split_lines))
for (i,line) in enumerate(split_lines)
  axonlengths[i] = parse(Float64,split(line[4], "=")[2])
end
calcAxonArea = (axonlengths .^ 2)./(4pi) #when computing axon area from length, it is assumed that the axons are circles
axonradius = sqrt.(calcAxonArea./pi) #in nm
# old method histogram
f = Figure()
minval,  maxval = extrema(calcAxonArea)
axh = Axis(f[1,1], xlabel = "Axon Area (nm^2)", title = "Sampling Axon Area (n=273)", xscale = log10)
hist!(axh, calcAxonArea, color = :purple, bins = exp.(LinRange(log(minval), log(maxval), 80)), normalization = :probability)
axradiusm = axonradius./1e9 #in m

#more accurate way of computing area using xy coordinates of contours
axonxcoord, axonycoord = read_contours("/Users/karrahhayes/Desktop/Research/xy_coords.txt")

axonarea_pixels = []
for i in 1:length(axonxcoord)
    A = 0.5 * abs(sum((axonycoord[i][j] + axonycoord[i][j+1]) * (axonxcoord[i][j] - axonxcoord[i][j+1]) for j in 1:length(axonxcoord[i])-1))
    push!(axonarea_pixels, A)
end
axonarea_nm = 5.102 * 5.102 .* axonarea_pixels
axonarea_um = axonarea_nm./1e6
axonradius_um = sqrt.(axonarea_um./pi)
axonradius_m = axonradius_um./1e6

#histogram with new area calculation method
n = Figure()
minval,  maxval = extrema(axonarea_um)
axn = Axis(n[1,1], xlabel = "Axon Area (um^2)", title = "Sampling Axon Area (n=273)", xscale = log10)
hist!(axn, axonarea_um, color = :blue, bins = exp.(LinRange(log(minval), log(maxval), 80)), normalization = :probability)
#Cm must be entered in F/m^2, Rm in ohm-m^2, and Ra in ohm-m
cv(Cm, Rm, Ra, axonradius_m) = (sqrt.((Rm .* axonradius_m)./(2 .* Ra)))./(Cm .* Rm)
axcv = Axis(n[1,2], xlabel = "Axon Area (m^2)", ylabel = "Estimated Conduction Velocity (m/sec)")
vx = axonarea_nm./1e18 #in m
vy = cv(0.01, 0.21, 1, axonradius_m)
scatter!(axcv, vx, vy, color = :green)
n

#=
#pull out coordinates of center of mass of axons
positionlines = readlines("/Users/karrahhayes/Desktop/Research/xy_coords.txt")
splitting = split.(positionlines[contains.(positionlines,"Center of Mass")], "(")
COMxcoord = zeros(length(splitting))
COMycoord = zeros(length(splitting))
for (k,line) in enumerate([split(split(s[2], ")")[1], ",") for s in splitting])
    COMxcoord[k] = parse(Float64, line[1])
    COMycoord[k] = parse(Float64, line[2])
end

#making plot with distribution of axon positions
p = Figure()
xmin, xmax = extrema(xcoord)
ymin, ymax = extrema(ycoord)
axp = Axis(p[1,1])
scatter!(axp, xcoord, ycoord, markersize = axonradius./2, markerspace = :data, color = axonradius)
p
=#

#way of assigning glia to its associated axon
glialines = readlines("/Users/karrahhayes/Desktop/Research/glia_info.txt")
matching_index = zeros(length())
#pull out glia xy coordinates same way as axons
#calculate centroid (mean of xcoords, mean of y coords) of each axon
#calculate centroid of each glia
#loop through axon centroid vector, compare each value in vector to a glia value and if it falls within a 
#certain error, then the glia gets assigned but if not it moves onto the next axon centroid
#assign by populating glia into new vector where indices align with axon #


#=
#getting info on glia
glialines = readlines("/Users/karrahhayes/Desktop/Research/axon and glia2.txt")
splitlines = split.(glialines[contains.(glialines, "CONTOUR")], ",")
glialengths = zeros(length(splitlines))
for (j, line) in enumerate(splitlines)
    glialengths[j] = parse(Float64,split(line[4], "=")[2])
end
calcGliaArea = (glialengths .^ 2)./(4pi)
gliaaxonradius = sqrt.(calcGliaArea./pi)

#calculating thickness
gliaThickness = gliaaxonradius - axonradius #does not work with current glia text file bc not all glia segmented

#adding linear trendline and 95% confidence interval
x = log10.(calcAxonArea)
y = log10.(gliaThickness)
function add_intercept_column(x::AbstractVector{T}) where {T}
    mat = similar(x, float(T), (length(x), 2))
    fill!(view(mat, :, 1), 1)
    copyto!(view(mat, :, 2), x)
    return mat
end
function fit_line!(axs, x, y)
    df = DataFrame(x=x, y=y)
    mod = lm(@formula(y ~ x), df)
    pred = GLM.predict(mod, add_intercept_column(df.x); interval=:confidence, level=0.95)
    inds = sortperm(df.x)
    band!(axs, df.x[inds], pred.lower[inds], pred.upper[inds];
        color=(:gray, 0.5),
        alpha=0.5)
    lines!(axs, df.x, pred.prediction, color=:gray, linewidth=7)
    return mod
end

#making scatterplot to determine correlation between axon area and glia thickness
g = Figure()
axs = Axis(c[1,1], xlabel = "Axon Area (nm^2)", ylabel = "Glia Thickness (nm)", title = "Relationship Between Axon Area and Surrounding Glia Thickness")
scatter!(axs, x, y)
fit_line!(axs, x, y)
g =#