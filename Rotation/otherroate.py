


// Projection 2D -> 3D matrix
        Mat A1 = (Mat_<double>(4,3) <<
            1, 0, -w/2,
            0, 1, -h/2,
            0, 0,    0,
            0, 0,    1);

// Rotation matrices around the X axis
        Mat R = (Mat_<double>(4, 4) <<
            1,          0,           0, 0,
            0, cos(alpha), -sin(alpha), 0,
            0, sin(alpha),  cos(alpha), 0,
            0,          0,           0, 1);

// Translation matrix on the Z axis 
        Mat T = (Mat_<double>(4, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, dist,
            0, 0, 0, 1);

// Camera Intrisecs matrix 3D -> 2D
        Mat A2 = (Mat_<double>(3,4) <<
            f, 0, w/2, 0,
            0, f, h/2, 0,
            0, 0,   1, 0);

Mat transfo = A2 * (T * (R * A1));

Mat source;
Mat destination;

warpPerspective(source, destination, transfo, source.size(), INTER_CUBIC | WARP_INVERSE_MAP);