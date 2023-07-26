import math
import numpy as np
from numba import jit, vectorize

SQRT2 = np.float32(2**0.5)
PI = np.float32(np.pi)


@jit
def positions(n, m, t, tref, v):
    """
    Calculates the (x, y) position of the m x n pixel grid at each time t.

    Inputs:
    n = height of grid, in pixels
    m = width of grid, in pixels
    t = array of times at which to output pixel grid positions
    tref = reference transit midpoint time [time units]
    v = grid velocity [1/(time units)] (since distance is in units of R*)

    Outputs:
        pos = array of shape (len(t), n, m, 2) containing the positions of pixel
        (n,m) at each time t. pos[k,:,:,0] are the x-positions of the whole
        grid at time t=k; pos[k,:,:,1] are the y-positions.
    """
    t_len = len(t)
    pos = np.zeros((t_len, n, m, 2), dtype=float)
    w = (2./n) # pixel height = diameter of stellar disk/number of pixels

    # y positions are constant in time
    # y = np.zeros(n)
    for i in range(n):
        # y[i] = 1. - (w/2.) - (i-1.)*w
        pos[:, i, :, 1] = 1. - (w/2.) - i*w

    # x positions are time-evolving
    jmid = 1. + (m-1.)/2.
    # at time t=tref:
    xmidref = 0. # x position of reference pixel at tref
    # pixels shift linearly in time

    if v != 0.:
        # tMin = tref - (2. + w*(m-1))/(2.*v)
        # tMax = tref + (2. + w*(m-1))/(2.*v)
        tMin = tref - (2. + w*m)/(2.*np.abs(v))
        tMax = tref + (2. + w*m)/(2.*np.abs(v))

        overlappingTimesMask = (t > tMin) & (t < tMax)
    else:
        overlappingTimesMask = np.ones_like(t).astype(bool)
    # print(overlappingTimesMask)
    overlappingTimes = t[overlappingTimesMask]
    overlappingPos = pos[overlappingTimesMask]

    for tind, time in enumerate(overlappingTimes):
        for j in range(m):
            xref = xmidref + (j+1-jmid)*w
            overlappingPos[tind, :, j, 0] = xref + (time - tref)*v

    # print(overlappingTimes)
    # print(overlappingPos)
    return overlappingPos, overlappingTimes


@jit
def chord_area(chord_length):
    carea = (math.asin(0.5*chord_length)
             - 0.5*math.sin(2.*math.asin(0.5*chord_length)))
    return carea


@jit
def numpy_sign(x):
    if x < 0:
        sign = -1
    elif x > 0:
        sign = 1
    elif x == 0:
        sign = 0
    else:
        sign = math.nan
    return sign


@jit
def overlap(x0, y0, w, verbose=False):
    """Calculate the overlap of a pixel and the stellar disk

    All measurements and units are relative to the stellar radius.
        i.e. the diameter of the star is 2, it's area pi, etc.
    args:
        x0 (float32) - x position of the pixel center
        y0 (float32) - y position of the pixel center
        w  (float32) - side length of the pixel, pixels are assumed square
    returns:
    """
    # radial distance to pixel location
    pix_r = (x0**2 + y0**2)**0.5

    if (pix_r >= 1+w/SQRT2) or (abs(x0) >= (1+w/2)) or (abs(y0) >= (1+w/2)):
        # pixel is guaranteed to be fully outside the star
        if verbose: print("Pixel outside")
        area = 0.0
        return area

    elif pix_r <= 1-w/SQRT2:
        # pixel is guaranteed to be fully inside the star
        if verbose: print("pixel inside")
        area = w**2
        return area/math.pi # normalize by stellar area, which is pi*1**2

    n_intersections = int(0)

    def test(x, y):
        return (abs(x-x0) <= w/2) and (abs(y-y0) <= w/2)

    # calculate coordinates of intersection solutions
    xarr = ((1. - 0.5*w - y0)*(1. + 0.5*w + y0))**0.5
    yarr = y0 + 0.5*w

    if test(xarr, yarr):
        if verbose: print("hit top 1")
        n_intersections += 1
        xg0 = xarr
        yg0 = yarr
        vg0 = 1

    xarr = -1.*xarr
    if test(xarr, yarr):
        if verbose: print("hit top 2")
        n_intersections += 1
        if n_intersections == 1:
            xg0 = xarr
            yg0 = yarr
            vg0 = 1
        else:
            xg1 = xarr
            yg1 = yarr
            vg1 = 1

    xarr = x0 + 0.5*w
    yarr = -1.*((1. - 0.5*w - x0)*(1. + 0.5*w + x0))**0.5
    if test(xarr, yarr):
        if verbose: print("hit right 1")
        n_intersections += 1
        if n_intersections == 1:
            xg0 = xarr
            yg0 = yarr
            vg0 = 2
        elif n_intersections == 2:
            xg1 = xarr
            yg1 = yarr
            vg1 = 2
        else:
            xg2 = xarr
            yg2 = yarr
            vg2 = 2

    yarr = -1.*yarr
    if test(xarr, yarr):
        if verbose: print("hit right 2")
        n_intersections += 1
        if n_intersections == 1:
            xg0 = xarr
            yg0 = yarr
            vg0 = 2
        elif n_intersections == 2:
            xg1 = xarr
            yg1 = yarr
            vg1 = 2
        elif n_intersections == 3:
            xg2 = xarr
            yg2 = yarr
            vg2 = 2
        else:
            xg3 = xarr
            yg3 = yarr
            vg3 = 2

    xarr = -1.*((1. + 0.5*w - y0)*(1. - 0.5*w + y0))**0.5
    yarr = y0 - 0.5*w
    if test(xarr, yarr):
        if verbose: print("hit bottom 1")
        n_intersections += 1
        if n_intersections == 1:
            xg0 = xarr
            yg0 = yarr
            vg0 = 3
        elif n_intersections == 2:
            xg1 = xarr
            yg1 = yarr
            vg1 = 3
        elif n_intersections == 3:
            xg2 = xarr
            yg2 = yarr
            vg2 = 3
        elif n_intersections == 4:
            xg3 = xarr
            yg3 = yarr
            vg3 = 3
        else:
            xg4 = xarr
            yg4 = yarr
            vg4 = 3

    xarr = -1.*xarr
    if test(xarr, yarr):
        if verbose: print("hit bottom 2")
        n_intersections += 1
        if n_intersections == 1:
            xg0 = xarr
            yg0 = yarr
            vg0 = 3
        elif n_intersections == 2:
            xg1 = xarr
            yg1 = yarr
            vg1 = 3
        elif n_intersections == 3:
            xg2 = xarr
            yg2 = yarr
            vg2 = 3
        elif n_intersections == 4:
            xg3 = xarr
            yg3 = yarr
            vg3 = 3
        elif n_intersections == 5:
            xg4 = xarr
            yg4 = yarr
            vg4 = 3
        else:
            xg5 = xarr
            yg5 = yarr
            vg5 = 3

    xarr = x0 - 0.5*w
    yarr = ((1. + 0.5*w - x0)*(1. - 0.5*w + x0))**0.5
    if test(xarr, yarr):
        if verbose: print("hit left 1")
        n_intersections += 1
        if n_intersections == 1:
            xg0 = xarr
            yg0 = yarr
            vg0 = 4
        elif n_intersections == 2:
            xg1 = xarr
            yg1 = yarr
            vg1 = 4
        elif n_intersections == 3:
            xg2 = xarr
            yg2 = yarr
            vg2 = 4
        elif n_intersections == 4:
            xg3 = xarr
            yg3 = yarr
            vg3 = 4
        elif n_intersections == 5:
            xg4 = xarr
            yg4 = yarr
            vg4 = 4
        elif n_intersections == 6:
            xg5 = xarr
            yg5 = yarr
            vg5 = 4
        else:
            xg6 = xarr
            yg6 = yarr
            vg6 = 4

    yarr = -1.*yarr
    if test(xarr, yarr):
        if verbose: print("hit left 2")
        n_intersections += 1
        if n_intersections == 1:
            xg0 = xarr
            yg0 = yarr
            vg0 = 4
        elif n_intersections == 2:
            xg1 = xarr
            yg1 = yarr
            vg1 = 4
        elif n_intersections == 3:
            xg2 = xarr
            yg2 = yarr
            vg2 = 4
        elif n_intersections == 4:
            xg3 = xarr
            yg3 = yarr
            vg3 = 4
        elif n_intersections == 5:
            xg4 = xarr
            yg4 = yarr
            vg4 = 4
        elif n_intersections == 6:
            xg5 = xarr
            yg5 = yarr
            vg5 = 4
        elif n_intersections == 7:
            xg6 = xarr
            yg6 = yarr
            vg6 = 4
        else:
            xg7 = xarr
            yg7 = yarr
            vg7 = 4

    if n_intersections == 2:
        # pixel partially overlaps with stellar disk
        if verbose: print("2 called")
        if vg0 == vg1:
            # points of intersection are both on the same face, e.g. for
            # the middle pixel of an odd-N pixel grid at first ingress
            area = 0. # will add chord correction later.

        elif vg0 % 2 == vg1 % 2:
            if verbose: print("tb or lr")
            # either the points of intersection are top/bottom sides,
            # or they are left/right
            # calculate overlap area as a trapezoid
            if vg0 % 2 == 1:
                if verbose: print("tb")
                # points of intersection are in top/bottom
                # distance from x = 0 to nearest vertical edge of box
                edgedist = x0 - (0.5*w)*numpy_sign(x0)
                a = abs(xg0-edgedist)
                b = abs(xg1-edgedist)
            else:
                # left/right
                # distance from y = 0 to nearest horizontal edge of box
                edgedist = y0 - (0.5*w)*numpy_sign(y0)
                a = abs(yg0-edgedist)
                b = abs(yg1-edgedist)
            area = 0.5*abs(a + b) * w

        else:
            # calculate overlap area as a triangle
            # if verbose: print("triangle")
            area = 0.5*abs(xg1 - xg0)*abs(yg1 - yg0)
            chord = math.hypot((xg1-xg0), (yg1-yg0))
            # 1**2 = (chord/2)**2 + (1-h)**2, so:
            h = 1. - (1. - (chord**2/4.))**0.5

            if math.hypot(x0, y0) < (1. - h):
                # most of pixel is inside the circle
                # if verbose: print("x0,y0: ({0},{1})".format(x0,y0))
                # if verbose: print("mostly inside")
                area = w**2 - area

        # append area with arc correction
        chord = ((xg1 - xg0)**2 + (yg1 - yg0)**2)**0.5
        area = (area + chord_area(chord))

    elif n_intersections == 3:
        if verbose: print("3 called")
        # possible when a pixel corner or edge skims the stellar radius
        # possibilities:
        #    left, top, right     = 4, 1, 2. verts_mod2 = 0, 1, 0
        #    left, bottom, right  = 4, 3, 2. verts_mod2 = 0, 1, 0
        #    bottom, left, top    = 3, 4, 1. verts_mod2 = 1, 0, 1
        #    bottom, right, top   = 3, 2, 1. verts_mod2 = 1, 0, 1

        # catch cases where pixel corner touches edge of stellar disk
        # (happens for pixels touching the midplane, if N is even)
        def ctest(x, y):
            return (abs(x) == 1.0 and (abs(y) == 0.0))

        if ctest(xg0, yg0):
            corners0 = 1
        else:
            corners0 = 0
        if ctest(xg1, yg1):
            corners1 = 1
        else:
            corners1 = 0
        if ctest(xg2, yg2):
            corners2 = 1
        else:
            corners2 = 0

        if corners0+corners1+corners2 == 3:
            area = 0.

        else:
            # verts_mod2 = [v % 2 for v in vg]
            verts_mod2_0 = vg0 % 2
            verts_mod2_1 = vg1 % 2
            verts_mod2_2 = vg2 % 2

            v0 = 0
            v1 = 0
            if vg0 == 0:
                v0 += 1
                v0_0 = 0
            elif vg0 == 1:
                v1 += 1
                v1_0 = 0
            if vg1 == 0:
                v0 += 1
                if v0 == 1:
                    v0_0 = 1
                else:
                    v0_1 = 1
            elif vg1 == 1:
                v1 += 1
                if v1 == 1:
                    v1_0 = 1
                else:
                    v1_1 = 1
            if vg2 == 0:
                v0 += 1
                if v0 == 1:
                    v0_0 = 2
                elif v0 == 2:
                    v0_1 = 2
            elif vg2 == 1:
                v1 += 1
                if v1 == 1:
                    v1_0 = 2
                elif v1 == 2:
                    v1_1 = 2

            if verts_mod2_0+verts_mod2_1+verts_mod2_2 == 1:
                # left, top, right or left, bottom, right
                middleidx0 = v1_0
                outeridx0 = v0_0
                outeridx1 = v0_1
                # outeridx = np.arange(3)[np.array(verts_mod2) == 0]

            elif verts_mod2_0+verts_mod2_1+verts_mod2_2 == 2:
                middleidx0 = v0_0
                # middleidx = np.arange(3)[np.array(verts_mod2) == 0]
                outeridx0 = v1_0
                outeridx1 = v1_1

            if middleidx0 == 0:
                midxg = xg0
                midyg = yg0
            elif middleidx0 == 1:
                midxg = xg1
                midyg = yg1
            else:
                midxg = xg2
                midyg = yg2

            if outeridx0 == 0:
                outerxg0 = xg0
                outeryg0 = yg0
            elif outeridx0 == 1:
                outerxg0 = xg1
                outeryg0 = yg1
            else:
                outerxg0 = xg2
                outeryg0 = yg2

            if outeridx1 == 0:
                outerxg1 = xg0
                outeryg1 = yg0
            elif outeridx1 == 1:
                outerxg1 = xg1
                outeryg1 = yg1
            else:
                outerxg1 = xg2
                outeryg1 = yg2

            area = w**2

            triarea = (0.5*abs(outerxg0 - midxg)
                       * abs(outeryg0 - midyg))
            chord = math.hypot((outerxg0-midxg), (outeryg0-midyg))
            h = 1. - (1. - (chord**2/4.))**0.5
            if ((x0**2 + y0**2)**0.5 < (1. - h)):
                # most of pixel is inside the circle
                triarea = -1*triarea

            area += triarea
            area += chord_area(chord)

            triarea = (0.5*abs(outerxg1 - midxg)
                       * abs(outeryg1 - midyg))
            chord = math.hypot((outerxg1-midxg), (outeryg1-midyg))
            h = 1. - (1. - (chord**2/4.))**0.5
            if ((x0**2 + y0**2)**0.5 < (1. - h)):
                # most of pixel is inside the circle
                triarea = -1*triarea

            area += triarea
            area += chord_area(chord)

    if n_intersections == 4:
        if verbose: print("4 called")
        # possibilities:
        #    left, top, top, right        = 4, 1, 1, 2
        #    left, bottom, bottom, right  = 4, 3, 3, 2
        #    bottom, left, left, top      = 3, 4, 4, 1
        #    bottom, right, right, top    = 3, 2, 2, 1

        # middlevert = Counter(vg).most_common(3)[0][0]
        if (vg0 == vg1):
            middlevert = vg0
            middlexg0 = xg0
            middlexg1 = xg1
            outerxg0 = xg2
            outerxg1 = xg3
            middleyg0 = yg0
            middleyg1 = yg1
            outeryg0 = yg2
            outeryg1 = yg3
        elif (vg0 == vg2):
            middlevert = vg0
            middlexg0 = xg0
            middlexg1 = xg2
            outerxg0 = xg1
            outerxg1 = xg3
            middleyg0 = yg0
            middleyg1 = yg2
            outeryg0 = yg1
            outeryg1 = yg3
        elif (vg0 == vg3):
            middlevert = vg0
            middlexg0 = xg0
            middlexg1 = xg3
            outerxg0 = xg1
            outerxg1 = xg2
            middleyg0 = yg0
            middleyg1 = yg3
            outeryg0 = yg1
            outeryg1 = yg2
        elif (vg1 == vg2):
            middlevert = vg1
            middlexg0 = xg1
            middlexg1 = xg2
            outerxg0 = xg0
            outerxg1 = xg3
            middleyg0 = yg1
            middleyg1 = yg2
            outeryg0 = yg0
            outeryg1 = yg3
        elif (vg1 == vg3):
            middlevert = vg1
            middlexg0 = xg1
            middlexg1 = xg3
            outerxg0 = xg0
            outerxg1 = xg2
            middleyg0 = yg1
            middleyg1 = yg3
            outeryg0 = yg0
            outeryg1 = yg2
        else:
            middlevert = vg2
            middlexg0 = xg2
            middlexg1 = xg3
            outerxg0 = xg0
            outerxg1 = xg1
            middleyg0 = yg2
            middleyg1 = yg3
            outeryg0 = yg0
            outeryg1 = yg1

        area = w**2  # full pixel, corrections following

        if middlevert % 2 == 1:
            # middle intersection is on top or bottom

            xd00 = abs(outerxg0 - middlexg0)
            xd01 = abs(outerxg0 - middlexg1)

            if xd00 <= xd01:
                xdshort = xd00
                mxg = middlexg1
                mygshort = middleyg0
                myg_long = middleyg1

            else:
                xdshort = xd01
                mxg = middlexg0
                mygshort = middleyg1
                myg_long = middleyg0

            triarea0 = 0.5*xdshort*abs(outeryg0 - mygshort)
            chord0 = math.hypot(xdshort, (outeryg0-mygshort))
            area = area - triarea0 + chord_area(chord0)

            triarea1 = (0.5 * abs(outerxg1 - mxg)
                        * abs(outeryg1 - myg_long))
            chord1 = math.hypot((outerxg1-mxg), (outeryg1-myg_long))
            area = area - triarea1 + chord_area(chord1)

        else:
            # middle intersection is on left or right
            yd00 = abs(outeryg0 - middleyg0)
            yd01 = abs(outeryg0 - middleyg1)
            if yd00 <= yd01:
                ydshort = yd00
                myg = middleyg1
                mxgshort = middlexg0
                mxg_long = middlexg1

            else:
                ydshort = yd01
                myg = middleyg0
                mxgshort = middlexg1
                mxg_long = middlexg0

            triarea0 = 0.5*ydshort*abs(outerxg0 - mxgshort)
            chord0 = (ydshort**2
                      + (outerxg0 - mxgshort)**2)**0.5
            area = (area - triarea0 + math.asin(0.5*chord0)
                    - 0.5*math.sin(2.*math.asin(0.5*chord0)))

            triarea1 = (0.5*abs(outerxg1 - mxg_long)
                        * abs(outeryg1 - myg))
            chord1 = ((outerxg1 - mxg_long)**2
                      + (outeryg1 - myg)**2)**0.5
            area = (area - triarea1 + math.asin(0.5*chord1)
                    - 0.5*math.sin(2.*math.asin(0.5*chord1)))

    if n_intersections == 6:
        if verbose: print("6 called")
        # This can happen when w>sqrt(2), w<2, abs(x0)<1-w/2, and abs(y0)<1-w/2
        # First, find which side didn't hit
        if ((vg0 != 1) and (vg1 != 1) and (vg2 != 1) and (vg3 != 1)
           and (vg4 != 1) and (vg5 != 1)):
            # top not hit
            rght_chord = abs(yg1-yg0)
            rght_carea = chord_area(rght_chord)
            bott_chord = abs(xg3-xg2)
            bott_carea = chord_area(bott_chord)
            left_chord = abs(yg5-yg4)
            left_carea = chord_area(left_chord)

            area = PI - (rght_carea + bott_carea + left_carea)

        elif ((vg0 != 2) and (vg1 != 2) and (vg2 != 2) and (vg3 != 2)
              and (vg4 != 2) and (vg5 != 2)):
            # right not hit
            top__chord = abs(yg1-yg0)
            top__carea = chord_area(top__chord)
            bott_chord = abs(xg3-xg2)
            bott_carea = chord_area(bott_chord)
            left_chord = abs(yg5-yg4)
            left_carea = chord_area(left_chord)

            area = PI - (top__carea + bott_carea + left_carea)
            pass
        elif ((vg0 != 3) and (vg1 != 3) and (vg2 != 3) and (vg3 != 3)
              and (vg4 != 3) and (vg5 != 3)):
            # bottom not hit
            top__chord = abs(yg1-yg0)
            top__carea = chord_area(top__chord)
            rght_chord = abs(xg3-xg2)
            rght_carea = chord_area(rght_chord)
            left_chord = abs(yg5-yg4)
            left_carea = chord_area(left_chord)

            area = PI - (top__carea + rght_carea + left_carea)
        else:
            # left not hit
            top__chord = abs(yg1-yg0)
            top__carea = chord_area(top__chord)
            rght_chord = abs(xg3-xg2)
            rght_carea = chord_area(rght_chord)
            bott_chord = abs(yg5-yg4)
            bott_carea = chord_area(left_chord)

            area = PI - (top__carea + rght_carea + bott_carea)
    if n_intersections == 8:
        if verbose: print("8 called")
        # intersections on all sides, w<2, w>sqrt2
        top__chord = abs(yg1-yg0)
        top__carea = chord_area(top__chord)
        rght_chord = abs(xg3-xg2)
        rght_carea = chord_area(rght_chord)
        bott_chord = abs(yg5-yg4)
        bott_carea = chord_area(left_chord)
        left_chord = abs(yg7-yg6)
        left_carea = chord_area(left_chord)
        area = PI - (top__carea + rght_carea + bott_carea + left_carea)
        pass
    norm_area = area/PI # normalize by stellar area, which is pi*1**2
    return norm_area


@vectorize(['float32(float32, float32, float32, boolean)'], target='cuda')
def overlap_gpu(x0, y0, w, verbose=False):
    return overlap(x0, y0, w, verbose)


# initialize the cuda implementations of the functions.
norm_area = overlap(0.96, 0.0, 0.1, verbose=False)
# Test 1 million pixels
xs = np.array(0.96*100, dtype=np.float32)
ys = np.zeros(100, dtype=np.float32)
ws = np.array([.1]*100, dtype=np.float32)
norm_area = overlap_gpu(xs, ys, ws, False)
p = positions(n=10, m=10, t=np.arange(10, dtype=float), tref=5., v=1.)
