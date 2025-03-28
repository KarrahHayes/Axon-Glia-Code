using GLMakie
using GLM
using DataFrames
using Statistics
using HypothesisTests

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
function fit_line(x, y)
    df = DataFrame(x=x, y=y)
    modd = lm(@formula(y ~ x), df)
    return modd
end


#IMPORTING TEXT FILE WITH NECK CONNECTIVE INFO


#populating coordinates of axons and glia into dictionary
coordlines = readlines("/Users/karrahhayes/Desktop/Research/sample_info.txt")
objectindices = findall(occursin.("OBJECT", coordlines))
objectnames = coordlines[findall(occursin.("OBJECT", coordlines)) .+ 1] 
output_dict = Dict()
objxcoord, objycoord = [], []
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
axondiameter_um = axonradius_um .* 2
largerthan10um = axondiameter_um[axondiameter_um .>= 10]
largerthan10um_position = [1268.33, 2126.67, 2537.33, 3150.33, 3970.67, 1501, 2301, 1253, 1205, 1925, 1349, 1722.33, 3079.61, 3007.6, 4369.74, 4741.78, 5407.84, 4579.76, 6607.96, 7567.57, 8119.63, 7657.58, 8635.68, 8329.65, 9403.75, 8791.69, 7969.61, 6086.72, 6650.78, 6992.81, 7832.9, 8366.95, 7664.88, 6386.75, 5876.7, 6410.75, 7334.85, 8129.76, 8681.82, 9119.86, 9701.92, 9233.87, 8897.84, 9785.93]

#area of glia AND axons
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
##


#VARIOUS MEASUREMENTS OF PHYSICAL CHARACTERISTICS OF NEURONS IN NECK CONNECTIVE


#distribution of axon diameter
d = Figure()
minval,  maxval = extrema(axondiameter_um)
axd = Axis(d[1,1], xlabel = "Axon Diameter (μm)", title = "Axon Size (n=371)", ylabel = "Probability", xscale = log10, xticks = [0.316, 1.0, 3.16, 10.0])
hist!(axd, axondiameter_um, color = RGBf(0.24, 0.7, 0.44), bins = exp.(LinRange(log(minval), log(maxval), 80)), normalization = :probability)
d
averagediameter = mean(axondiameter_um)
logSDdiameter = std(log10.(axondiameter_um))

#spatial distribution of axons larger than 10 um
s = Figure() 
axs = Axis(s[1,1], yticklabelsvisible = false, xlabel = "Number of Axons With Diameters Over 10 μm", ylabel = "Position in Connective")
#hideydecorations!(axs, ticks = false)
hist!(axs, largerthan10um_position, bins = 15, color = RGBf(0.24, 0.7, 0.44), direction = :x)
s
##
#axon circularity
c = Figure()
axc = Axis(c[1,1], xlabel = "Log of Axon Radius (um)", ylabel = "Axon Circularity", title = "Axon Size vs. Axon Circularity")
ylims!(axc, (0,1))
scatter!(axc, log10.(axonradius_um), axonroundness, color = :red)
c

#distribution of axon positions
p = Figure()
xmin, xmax = extrema(axonCOM_x)
ymin, ymax = extrema(axonCOM_y)
axp = Axis(p[1,1], title = "Axon Positions", xlabel = "X-Coordinate (px)", ylabel = "Y-Coordinate (px)")
scatter!(axp, axonCOM_x, axonCOM_y, markersize = axonradius_pixels.*2, markerspace = :data, color = axonradius_pixels)
p

#mitochondria count in transverse image
mitolines = readlines("/Users/karrahhayes/Desktop/Research/mitochondria_sample.txt")
splitlines = split.(mitolines[contains.(mitolines, "CONTOUR")], "0  ")
mitocount = zeros(length(splitlines))
for (i,line) in enumerate(splitlines)
    mitocount[i] = parse(Float64,split(line[2], " points")[1])
end
mitox_ind = []
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
mitocoeffs = fit_line(purged_mx, purged_my)
##


#EVALUATING RELATIONSHIP BETWEEN AXON SIZE AND ENSHEATHING GLIA  


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

#calculating thickness
gliathickness_um = []
assoc_axon_radius = []
for (i, line) in enumerate(matchedaxon)
    gliathickness_um = abs.(gliaradius_um .- axonradius_um[line])
    push!(assoc_axon_radius, axonradius_um[line])
end
assoc_axon_diameter = assoc_axon_radius .* 2

#g-ratio plot
gratio = (assoc_axon_radius .* 2)./(gliaradius_um .* 2)
gr = Figure()
axgr = Axis(gr[1,1], xlabel = "Fiber Diameter (um)", ylabel = "g-Ratio", title = "Axon Radius vs. g-Ratio")
ylims!(axgr, (0,1))
scatter!(axgr, gliaradius_um .* 2, gratio, color = :magenta)
gr

#correlation between axon radius and glia thickness
update_theme!(fontsize=20)
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
axg = Axis(g[1,1], xlabel = "Log of Axon Radius (μm)", ylabel = "Log of Glia Thickness (μm)", title = "Axon Size vs. Surrounding Glia Thickness (n=137)")
scatter!(axg, ggx, ggy, color = RGBf(0.24, 0.7, 0.44))
#fit_line!(axg, ggx, ggy)
g 
gliacoeffs = fit_line(gx, gy) 
#note that these coefficients are ONLY for axons with radii greater than 1 um
#if you want the coefficients for all axon-glia pairs, then do fit_line(ggx, ggy)

#correlation between axon diameter and glia thickness
update_theme!(fontsize=20)
gdgx = log10.(assoc_axon_diameter)
gd_ind = []
for i = 1:length(gdgx)
    if gdgx[i] > 0.30102999566
        push!(gd_ind, i)
    end
end
gdx = gdgx[gd_ind]
gdy = log10.(gliathickness_um)[gd_ind]
gdgy = log10.(gliathickness_um)
gd = Figure()
axgd = Axis(gd[1,1], xlabel = "Log of Axon Diameter (μm)", ylabel = "Log of Glia Thickness (μm)", title = "Axon Size vs. Surrounding Glia Thickness (n=137)")
scatter!(axgd, gdgx, gdgy, color = RGBf(0.24, 0.7, 0.44))
#fit_line!(axg, ggx, ggy)
gd 
gliacoeffs = fit_line(gdx, gdy) 
#note that these coefficients are ONLY for axons with diameters greater than 2 um
#if you want the coefficients for all axon-glia pairs, then do fit_line(gdgx, gdgy)


##


# DETERMINING WHETHER NEURONS CAN BE CLASSIFIED AS MYELINATED OR NOT


#estimating with Pearson paper (https://doi.org/10.1242/jeb.53.2.299)
moth_calculated_CV = [-4.283972431663765,-4.363633674562362,-4.331358383811586,3.3333366402149256,2.466259261008756,1.6143502859787362,1.2307694411571704,1.9764714851215144,1.846154567484078,-14.117616213611452,2.2646296748810957,2.2350343496841623,2.2888900393891736,-4.799996342859937,2.221429790444466,2.8284981366104276,1.6744191053080413,2.2500011444811023,8.92862722743023,3.9406878257725966,-4.830408440063775,-6.77824051616282,-4.364454263658649,-4.044331919152116,-3.015163409192404,-3.446172977209739,-2.621811989795271]
#these were taken from df.velocity in julia file for CV code
allmothCV_madepositive = [4.283972431663765,4.363633674562362,4.331358383811586,3.3333366402149256,2.466259261008756,1.6143502859787362,1.2307694411571704,1.9764714851215144,1.846154567484078,14.117616213611452,2.2646296748810957,2.2350343496841623,2.2888900393891736,4.799996342859937,2.221429790444466,2.8284981366104276,1.6744191053080413,2.2500011444811023,8.92862722743023,3.9406878257725966,4.830408440063775,6.77824051616282,4.364454263658649,4.044331919152116,3.015163409192404,3.446172977209739,2.621811989795271]

roachdiameter_um = [0.20402170135737557, 0.34658250495297643, 0.8369230236148567, 0.8526933919820637, 1.0894379199873796, 1.2846699395054788]
roachCV_m_s = [-0.28872962663275925, -0.3220514006087928, 0.23424652256001233, 0.21807883481678414, 0.4273340268947712, 0.5722729549570378]
roachmodel = fit_line(roachdiameter_um, roachCV_m_s)
roachyint = coef(roachmodel)[1]
# power law: y = b*x^a or log10(y) = alog10(x) + log10(b) in log form 
# just found that yint = log10(b) 
# therefore, log10(diameter) = a*log10(CV) + yint
# can solve for diameter by undoing the log: estimated diameter = 10^(0.78*log10(CV) + yint)
# and thus, estimated CV = 10^yint * diameter^0.78
# for roach, a = 0.78
estdiameterfromroach = 10 .^((log10.(allmothCV_madepositive) .- roachyint) ./ 0.78)
estCVfromroach = (10 .^ roachyint) .* (largerthan10um .^ 0.78)

# comparing whether estimated CVs could be produced by measured diameters and vice versa using Kolmogorov-Smirnov (KS) test
# want to convert to log space to do KS test
usingdiameter = ApproximateTwoSampleKSTest(log10.(largerthan10um), log10.(estdiameterfromroach))
usingCV = ApproximateTwoSampleKSTest(log10.(allmothCV_madepositive), log10.(estCVfromroach))
##
#=
locustdiameter_um = [0.3036960226912455,0.30279215597694864,0.33322739903735005,0.3294144486963664,0.35250482526327936,0.3568494619067906,0.3684706053763225,0.3975158685315446,0.3924724441761399,0.39329276018575393,0.37571673012791296,0.42399384639741916,0.41927702934213856,0.4229228782737564,0.43893423149844474,0.45005407073987913,0.45802936527779314,0.4678351798001237,0.4857150306022662,0.5066634709218536,0.4879481130728821,0.46222968706776124,0.45679889126337214,0.47151900632477917,0.48860892319173777,0.49476888878245046,0.5000325831774737,0.49473091118941276,0.5217101932833847,0.5240951861261515,0.5256446719220891,0.5270346518272684,0.5206999893085823,0.5410180015837442,0.5562546119104638,0.567427619782151,0.5485299694865985,0.5507478609199994,0.5472387313233171,0.5645641092671094,0.567328878040253,0.578653796284091,0.5923029432218352,0.5848745260236639,0.60095423891582,0.6173301770336701,0.5952651954787747,0.612393089938771,0.6168516593613953,0.6204139575883303,0.6318679996484963,0.6410661726822238,0.6494592207435523,0.6612094880294124,0.6690556587509982,0.6813983764882461,0.6932094079229665,0.6809274543345788,0.7081118154309543,0.7207735449497188,0.7450184403449773,0.7395268803917281,0.8317516673244443,0.8457578036367428,0.8450590159248494,0.8336809290507587,0.8787907140609219,0.9201559083975693,0.9270450437746055,0.9270450437746055,0.9388332886535032,0.9047825787359142,0.8869027279337717,0.9249258940831027,0.9330834810675976,0.9598956617522038,1.0089627119568938,1.0261817526401806,1.0203863719426298,1.0203863719426298,1.006304280444256,0.9724434584918552,1.007436012716779,1.0084310256543663,1.061576869351304,1.0754614773658817,1.0982860107815309,1.1008229139964483,1.0853660336301103,1.0871509805028814,1.0871509805028814]
locustCV_m_s = [-0.18004180677996906,-0.1253900451392045,-0.1533369687055045,-0.09071768305613614,-0.0870217213487231,-0.1729982731982187,-0.1379623738980278,-0.09903359689781577,-0.07341937047471903,-0.017631555030446222,-0.013799266866612148,0.02213032809233839,-0.013284255809021706,-0.06236178012057314,-0.08776394316701508,-0.10686479444999841,-0.13814414250658913,-0.09033899845496685,-0.069238692477809,-0.08143233663546301,-0.042367233178829955,-0.03347571874337296,0.0016359174770517182,0.02681086976279201,0.06907207125329462,0.04918355599987878,0.02603835317640657,0.0041958253809566415,0.08843042806507317,0.0638916659092974,0.02718955436396131,0.000015147384046843015,-0.02699263837135324,-0.02141840104213999,-0.0020146020782210172,0,0.03877730315974426,0.06913266078948177,-0.05539398345905655,-0.05539398345905655,0.12022478717925411,0.12223938925747535,0.1250568026901755,0.0615892635341877,0.0571662273925293,0.06302826501863135,0.023584476960828882,0.03582356327062319,-0.000015147384046731993,0.1651064861098488,0.1457329819140234,0.12729861552909805,0.15406404313974975,0.1484898058105366,0.19965766912054295,0.20821594110697084,0.2185464570268716,0.1571692568693388,0.18061740737374654,0.24750825532430554,0.24040413220636792,0.20930655275833865,0.30325062861643803,0.24918961495349756,0.2181829198097489,0.1990517737586719,0.3123845011966433,0.34978339240813106,0.32998576145899605,0.32998576145899605,0.3477990850980035,0.2705928686115908,0.24876548820018796,0.2510375958072042,0.21197249235057103,0.2458268956951135,0.2475688448604927,0.2752279681299039,0.3488896967493713,0.38397103820170253,0.3763216092580812,0.3742767124117665,0.42497500681632294,0.45204338210790995,0.4269744615104971,0.46564573298191414,0.4513617498258051,0.4253839861855857,0.42730770395952633,0.38268351055772665,0.35101033051592]
locustmodel = fit_line(locustdiameter_um, locustCV_m_s)
locustyint = coef(locustmodel)[1]
# for locust, a = 0.7
estdiameterfromlocust = 10 .^((log10.(allmothCV_madepositive) .- locustyint) ./ 0.7)
estCVfromlocust = (10 .^ roachyint) .* (largerthan10um .^ 0.7)
=#

#=
#figure in the linear space
myelinornot = Figure()
estdiamin, estdiamax = extrema(estdiameterfromroach)
measdiamin, measdiamax = extrema(largerthan10um)
estCVmin, estCVmax = extrema(estCVfromroach)
measCVmin, measCVmax = extrema(allmothCV_madepositive)
myelindiameteraxis = Axis(myelinornot[1,1], xlabel = "Diameter (μm)", ylabel = "Probability", xticks = [0, 25, 50, 75, 100])
xlims!(myelindiameteraxis, 0, 100)
myelinCVaxis = Axis(myelinornot[2,1], xlabel = "Conduction Velocity (m/s)", ylabel = "Probability", xticks = [0, 2.5, 5, 7.5, 10])
xlims!(myelinCVaxis, 0, 10)
estdiaroach = hist!(myelindiameteraxis, estdiameterfromroach, bins = LinRange(estdiamin, estdiamax, 80), color = :orange, normalization = :probability)
measdiamoth = hist!(myelindiameteraxis, largerthan10um, bins = LinRange(measdiamin,measdiamax, 7), color = :blue, normalization = :probability)
hist!(myelinCVaxis, estCVfromroach, bins = LinRange(estCVmin, estCVmax, 7), color = :orange, normalization = :probability)
hist!(myelinCVaxis, allmothCV_madepositive, bins = LinRange(measCVmin, measCVmax, 80), color = :blue, normalization = :probability)
Legend(myelinornot[1,2], [estdiaroach, measdiamoth], ["Estimated Values", "Measured Values"])
myelinornot =#

#figure in the log space
myelinornot = Figure()
estdiamin, estdiamax = extrema(log10.(estdiameterfromroach))
measdiamin, measdiamax = extrema(log10.(largerthan10um))
estCVmin, estCVmax = extrema(log10.(estCVfromroach))
measCVmin, measCVmax = extrema(log10.(allmothCV_madepositive))
myelindiameteraxis = Axis(myelinornot[1,1], xlabel = "Log of Diameter (μm)", ylabel = "Probability")
#xlims!(myelindiameteraxis, 0, 100)
myelinCVaxis = Axis(myelinornot[2,1], xlabel = "Log of Conduction Velocity (m/s)", ylabel = "Probability")
#xlims!(myelinCVaxis, 0, 10)
estdiaroach = hist!(myelindiameteraxis, log10.(estdiameterfromroach), bins = LinRange(estdiamin, estdiamax, 50), color = :red, normalization = :probability)
measdiamoth = hist!(myelindiameteraxis, log10.(largerthan10um), bins = LinRange(measdiamin,measdiamax, 10), color = :blue, normalization = :probability)
estCVroach = hist!(myelinCVaxis, log10.(estCVfromroach), bins = LinRange(estCVmin, estCVmax, 10), color = :orange, normalization = :probability)
measCVmoth = hist!(myelinCVaxis, log10.(allmothCV_madepositive), bins = LinRange(measCVmin, measCVmax, 50), color = :green, normalization = :probability)
Legend(myelinornot[1,2], [estdiaroach, measdiamoth], ["Estimated Diameters using Pearson, et al., 1970", "Measured Diameters of M. sexta"])
Legend(myelinornot[2,2], [estCVroach, measCVmoth], ["Estimated Conduction Velocities using Pearson, et al., 1970", "Measured Conduction Velocities of M. sexta"])
myelinornot
##


# ATTEMPTING TO ESTIMATE CONDUCTION VELOCITY FROM ELECTRICAL CONSTANTS

#estimating CV-axon diameter curve
axondiameter_m = axondiameter_um./1e6
n = Figure()
#Cm must be entered in F/m^2, Rm in ohm-m^2, and Ra in ohm-m
cv(Cm, Rm, Ra, axonradius_m) = (sqrt.((Rm .* axonradius_m)./(2 .* Ra)))./(Cm .* Rm)
axcv = Axis(n[1,1], xlabel = "Axon Diameter (m)", ylabel = "Estimated Conduction Velocity (m/sec)")
vx = axondiameter_m
vy = cv(0.01, 0.21, 1, axonradius_m)
scatter!(axcv, vx, vy, color = :green)
n
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