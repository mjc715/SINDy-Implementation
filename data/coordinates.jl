const EARTH_RADIUS = 6371.0 

struct EquirectangularReference{T<:AbstractFloat}
    lon0::T
    lat0::T
    R::T

    function EquirectangularReference(; lon0::Real = -75.0, lat0::Real = 10.0, R::Real = EARTH_RADIUS)
        @assert -180.0 <= lon0 <= 180.0 "The longitude must be between -180 degrees and 180 degrees."
        @assert -90 <= lat0 <= 90 "The latitude must be between -90 degrees and 90 degrees."
    
        lon0, lat0, R = promote(float(lon0), float(lat0), float(R))
    
        return new{typeof(lon0)}(lon0, lat0, R)
    end
end

function sph2xy(lon::Real, lat::Real, eqr::EquirectangularReference)
    @assert -180.0 <= lon <= 180.0 "The longitude must be between -180 degrees and 180 degrees."
    @assert -90 <= lat <= 90 "The latitude must be between -90 degrees and 90 degrees."

    lon0, lat0, R = (eqr.lon0, eqr.lat0, eqr.R)
    deg2rad = π/180

    x = R*(lon - lon0)*deg2rad*cos(lat0*deg2rad)
    y = R*(lat - lat0)*deg2rad

    return [x, y]
end

function sph2xy(lon_range::AbstractRange, lat_range::AbstractRange, eqr::EquirectangularReference)
    # uses the fact that the translation between eqr and spherical is linear
    lonmin, latmin = sph2xy(first(lon_range), first(lat_range), eqr)
    lonmax, latmax = sph2xy(last(lon_range), last(lat_range), eqr)

    return  (
            range(start = lonmin, length = length(lon_range), stop = lonmax), 
            range(start = latmin, length = length(lat_range), stop = latmax)
            )
end

function sph2xy(lon_lat::Matrix{T}, eqr::EquirectangularReference) where {T<:Real}
    @assert size(lon_lat, 2) == 2 "lon_lat should be an `N x 2` matrix"
    xy = zeros(T, size(lon_lat))
    
    for i = 1:size(lon_lat, 1)
        xy[i,:] .= sph2xy(lon_lat[i,1], lon_lat[i,2], eqr)
    end

    return xy
end

function sph2xy(lon_lat::Vector{T}, eqr::EquirectangularReference) where {T<:Real}
    @assert iseven(length(lon_lat)) "lon_lat should be of the form `[lon1, lat1, lon2, lat2 ... lon3, lat3]`."

    xy = zeros(T, length(lon_lat))
    
    for i = 1:2:length(lon_lat)
        xy[i:i+1] .= sph2xy(lon_lat[i], lon_lat[i + 1], eqr)
    end

    return xy
end

function xy2sph(x::Real, y::Real, eqr::EquirectangularReference)
    lon0, lat0, R = (eqr.lon0, eqr.lat0, eqr.R)
    deg2rad = π/180
    rad2deg = 1/deg2rad

    lon = lon0 + rad2deg*x/(R*cos(lat0*deg2rad))
    lat = lat0 + rad2deg*y/R 

    return [lon, lat]
end

function xy2sph(xy::Vector{<:Vector{T}}, eqr::EquirectangularReference) where {T<:Real}
    lonlat = zeros(T, length(xy), 2) 
    
    for i = 1:length(xy)
        lonlat[i,:] = xy2sph(xy[i][1], xy[i][2], eqr)
    end

    return lonlat
end

function xy2sph(xy::Matrix{T}, eqr::EquirectangularReference) where {T<:Real}
    @assert size(xy, 2) == 2 "xy should be an `N x 2` matrix"
    lonlat = zeros(T, size(xy))
    
    for i = 1:size(xy, 1)
        lonlat[i,:] = xy2sph(xy[i,1], xy[i,2], eqr)
    end

    return lonlat
end

function xy2sph(x_range::AbstractRange, y_range::AbstractRange, eqr::EquirectangularReference)
    # uses the fact that the translation between eqr and spherical is linear
    xmin, ymin = xy2sph(first(x_range), first(y_range), eqr)
    xmax, ymax = xy2sph(last(x_range), last(y_range), eqr)

    return  (
            range(start = xmin, length = length(x_range), stop = xmax), 
            range(start = ymin, length = length(y_range), stop = ymax)
            )
end

function xy2sph(xy::Vector{T}, eqr::EquirectangularReference) where {T<:Real}
    @assert iseven(length(xy)) "xy should be of the form `[x1, y1, x2, y2 ... x3, y3]`."

    lon_lat = zeros(T, length(xy))
    
    for i = 1:2:length(xy)
        lon_lat[i:i+1] .= xy2sph(xy[i], xy[i + 1], eqr)
    end

    return lon_lat
end