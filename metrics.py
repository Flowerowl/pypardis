def geo_distance(p1, p2):
    """ p1(lat1, lon1)
        p2(lat2, lon2)
    """
    lat1, lon1 = p1
    lat2, lon2 = p2

    theta = lon1 -lon2
    dist = math.sin(math.radians(lat1))*math.sin(math.radians(lat2)) \
            + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))\
            * math.cos(math.radians(theta))
    if dist - 1 > 0 :
        dist = 1
    elif dist +1 < 0 :
        dist = -1
    dist = math.acos(dist)
    dist = math.degrees(dist)
    miles = dist * 60 * 1.1515 * 1.609344

    return miles
