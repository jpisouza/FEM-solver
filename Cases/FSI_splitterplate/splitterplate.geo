// Gmsh project created on Tue Jun 22 11:18:11 2021
SetFactory("OpenCASCADE");
f = 0.8;
//+
Point(1) = {-3, -3, 0, f};
//+
Point(2) = {10, -3, 0, 0.2*f};
//+
Point(3) = {10, 3, 0, 0.2*f};
//+
Point(4) = {-3, 3, 0, f};

//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};


//+
Point(5) = {0.989897948, 0.1, 0, 0.1*f};
//+
Point(6) = {0.989897948, -0.1, 0, 0.1*f};
//+
Point(7) = {4, -0.1, 0, 0.1*f};
//+
Point(8) = {4, 0.1, 0, 0.1*f};

//+
Point(9) = {0.5, 0, 0, 0.1*f};
//+
Point(10) = {0.5, 0.5, 0, 0.1*f};

//+
Point(11) = {0, 0, 0, 0.1*f};
//+
Point(12) = {0.5, -0.5, 0, 0.1*f};
//+
Line(5) = {6, 7};
//+
Line(6) = {7, 8};
//+
Line(7) = {8, 5};
//+
Line(8) = {5, 6};
//+
Circle(9) = {5, 9, 10};
//+
Circle(10) = {10, 9, 11};
//+
Circle(11) = {11, 9, 12};
//+
Circle(12) = {12, 9, 6};
//+
Curve Loop(1) = {7, 8, 5, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {4, 1, 2, 3};
//+
Curve Loop(3) = {9, 10, 11, 12, 5, 6, 7};
//+
Plane Surface(2) = {2, 3};
//+
Physical Curve("inFlow", 13) = {4};
//+
Physical Curve("outFlow", 14) = {2};
//+
Physical Curve("top", 15) = {3};
//+
Physical Curve("bottom", 16) = {1};
//+
Physical Curve("cylinder", 17) = {9, 10, 11, 12};
//+
Physical Curve("plate", 18) = {7, 6, 5};
//+
Physical Surface("solid", 19) = {1};
//+
Physical Surface("fluid", 20) = {2};
//+
Physical Curve("fixed", 21) = {8};
