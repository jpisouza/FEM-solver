// Gmsh project created on Sun Aug 01 16:16:26 2021
SetFactory("OpenCASCADE");
//+
f=0.07;
Point(1) = {-3, -0.5, 0, f};
//+
Point(2) = {0, -0.5, 0, f};
//+
Point(3) = {0, 0.5, 0, f};
//+
Point(4) = {-3, 0.5, 0, f};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Point(5) = {3, -0.5, 0, f};
//+
Point(6) = {3, 0.5, 0, f};
//+
Line(5) = {2, 5};
//+
Line(6) = {5, 6};
//+
Line(7) = {6, 3};
//+

//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Plane Surface(1) = {1};
//+
Physical Curve("in_flow", 8) = {4};
//+
Physical Curve("out_flow", 9) = {6};
//+
Physical Curve("wall", 10) = {3, 7, 1, 5};
//+
Physical Surface("free", 11) = {1};

//+
Curve Loop(2) = {7, -2, 5, 6};
//+
Plane Surface(2) = {2};
//+
Physical Surface("porous", 12) = {2};
