///////////////////////////////////////////////////////////////////////////////
// Basic variables
// Defined only if not given on the command line as 
//     gmsh -setnumber l2 5 -setnumber h2 3 ...
If (!Exists(h2)) h2 =   1; EndIf
If (!Exists(l2)) l2 =   4; EndIf
If (!Exists( d))  d = 0.3; EndIf
If (!Exists( f))  f =   2; EndIf
If (!Exists( h))  h = 0.1; EndIf

// Secondary variables
r = d/2;
c = d*f;
Lc1 = h;
Lc2 = h/2;

///////////////////////////////////////////////////////////////////////////////
// Points

// External boundaries
Point(1) = { 0, -h2/2, 0, Lc1};
Point(2) = {l2, -h2/2, 0, Lc1};
Point(3) = {l2,  h2/2, 0, Lc1};
Point(4) = { 0,  h2/2, 0, Lc1};

// Circle center + points at angles 180, 270, 0, 90 degrees
Point(5) = {c + 0,  0, 0, Lc2};
Point(6) = {c - r,  0, 0, Lc2};
Point(7) = {c + 0, -r, 0, Lc2};
Point(8) = {c + r,  0, 0, Lc2};
Point(9) = {c + 0,  r, 0, Lc2};

///////////////////////////////////////////////////////////////////////////////
// Lines

// Straight lines on the external boundary
Line(1)  = {1,2}; 
Line(2)  = {2,3};
Line(3)  = {3,4};
Line(4)  = {4,1};

// Circle segments {starting at, center at, ending at}
Circle(5) = {6,5,7};
Circle(6) = {7,5,8};
Circle(7) = {8,5,9};
Circle(8) = {9,5,6};

///////////////////////////////////////////////////////////////////////////////

// Line loops

Line Loop(1)  = {1,2,3,4};
Line Loop(2) = {5,6,7,8};

// Surfaces, the first is the outer line loop, the following are the hole line loops

Plane Surface(1) = {1,2};

///////////////////////////////////////////////////////////////////////////////
// Physical lines (for use with boundary conditions)

Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Line(5) = {5,6,7,8};

// Physical Surface - dolfin convert requires this for some reason

Physical Surface(0) = {1};
