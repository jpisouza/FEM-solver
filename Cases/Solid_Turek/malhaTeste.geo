// Gmsh project created on Mon Oct 05 18:24:42 2020

SetFactory("OpenCASCADE");
fator=0.005;
//+
Point(1) = {0, 0, 0, fator};
//+
Point(2) = {0.35, 0, 0, fator};
//+
Point(3) = {0.35, 0.02, 0, fator};
//+
Point(4) = {0, 0.02, 0, fator};
//+
Line(1) = {3, 4};
//+
Line(2) = {2, 3};
//+
Line(3) = {1, 2};


//+
Point(5) = {-0.048989794, 0.01, 0, 1.0};
//+
Circle(4) = {1, 5, 4};
//+
Curve Loop(1) = {1, -4, 3, 2};
//+
Plane Surface(1) = {1};
//+
Physical Curve("upper_bound", 5) = {1};
//+
Physical Curve("lower_bound", 6) = {3};
//+
Physical Curve("left_bound", 7) = {4};
//+
Physical Curve("right_bound", 8) = {2};
//+
Physical Surface("surface", 9) = {1};
