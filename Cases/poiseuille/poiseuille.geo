// Gmsh project created on Tue Jun 22 11:18:11 2021
SetFactory("OpenCASCADE");
f = 0.07;
//+
Point(1) = {-3, -0.5, 0, f};
//+
Point(2) = {3, -0.5, 0, f};
//+
Point(3) = {3, 0.5, 0, f};
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
Physical Line("inFlow") = {4};
//+
Physical Line("outFlow") = {2};
//+
Physical Line("wall") = {3, 1};
//+
Line Loop(1) = {3, 4, 1, 2};
//+
Plane Surface(1) = {1};
//+
Physical Surface("surface") = {1};
