// Gmsh project created on Mon Mar 08 19:31:47 2021
SetFactory("OpenCASCADE");
//+
f = 0.1;
Point(1) = {0, -0.5, 0, 0.6*f};
//+
Point(2) = {2, -0.5, 0, 0.6*f};
//+
Point(3) = {2, 0.5, 0, 0.6*f};
//+
Point(4) = {0, 0.5, 0, 0.6*f};
//+
Point(5) = {3, -2, 0, f};
//+
Point(6) = {3, 2, 0, f};
//+
Point(7) = {7, -2, 0, f};
//+
Point(8) = {7, 2, 0, f};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 5};
//+
Line(4) = {5, 7};
//+
Line(5) = {7, 8};
//+
Line(6) = {8, 6};
//+
Line(7) = {6, 3};
//+
Line(8) = {3, 4};
//+
Line Loop(1) = {8, 1, 2, 3, 4, 5, 6, 7};
//+
Plane Surface(1) = {1};
//+
Physical Line("inFlow") = {1};
//+
Physical Line("outFlow") = {5};
//+
Physical Line("wall") = {6, 7, 8, 2, 3, 4};
//+
Physical Surface("nozzle") = {1};

