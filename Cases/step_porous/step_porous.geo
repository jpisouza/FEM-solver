// Gmsh project created on Sun Aug 01 16:16:26 2021
SetFactory("OpenCASCADE");
//+
f=0.3;
Point(1) = {-25, -2, 0, f};
//+
Point(2) = {25, -2, 0, f};
//+
Point(3) = {25, 2, 0, f};
//+
Point(4) = {-25, 2, 0, f};
//+
Point(5) = {-25, -5, 0, f};
//+
Point(6) = {25, -5, 0, f};
//+
Line(1) = {4, 3};
//+
Line(2) = {3, 2};
//+
Line(3) = {2, 6};
//+
Line(4) = {5, 6};
//+
Line(5) = {5, 1};
//+
Line(6) = {1, 4};
//+
Line(7) = {1, 2};
//+
Physical Curve("wall", 8) = {1};
//+
Physical Curve("in_flow", 9) = {6};
//+
Physical Curve("out_flow", 10) = {2, 3};
//+
Physical Curve("slip_left", 11) = {5};
//+
Physical Curve("slip_bottom", 12) = {4};

//+
Curve Loop(1) = {7, -2, -1, -6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {4, -3, -7, -5};
//+
Plane Surface(2) = {2};
//+
Physical Surface("free", 13) = {1};
//+
Physical Surface("porous", 14) = {2};
