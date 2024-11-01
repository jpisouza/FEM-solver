// Gmsh project created on Tue Jun 22 11:18:11 2021
SetFactory("OpenCASCADE");
f = 1;
//+
Point(1) = {-3, -5, 0, f};
//+
Point(2) = {10, -5, 0, 0.1*f};
//+
Point(3) = {10, 5, 0, 0.1*f};
//+
Point(4) = {-3, 5, 0, f};

//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Point(5) = {0, 0, 0, 1};
//+
Point(6) = {0.5, 0, 0, 0.07*f};
//+
Point(7) = {0, 0.5, 0, 0.07*f};
//+
Point(8) = {-0.5, 0, 0, 0.07*f};
//+
Point(9) = {0, -0.5, 0, 0.07*f};
//+
Circle(5) = {9, 5, 6};
//+
Circle(6) = {6, 5, 7};
//+
Circle(7) = {7, 5, 8};
//+
Circle(8) = {8, 5, 9};

//+
Curve Loop(1) = {7, 8, 5, 6};
//+
Plane Surface(1) = {1};
//+
Physical Curve("cylinder", 9) = {7, 6, 5, 8};
//+
Physical Surface("solid", 10) = {1};
//+
Curve Loop(2) = {3, 4, 1, 2};
//+
Curve Loop(3) = {7, 8, 5, 6};
//+
Plane Surface(2) = {2, 3};
//+
Physical Surface("fluid", 11) = {2};
//+
Physical Curve("top", 12) = {3};
//+
Physical Curve("inFlow", 13) = {4};
//+
Physical Curve("outFlow", 14) = {2};
//+
Physical Curve("bottom", 15) = {1};
