// Gmsh project created on Tue Jun 22 11:18:11 2021
SetFactory("OpenCASCADE");
f = 0.2;
//+
Point(1) = {0, 0, 0, f};
//+
Point(2) = {5, 0, 0, 0.3*f};
//+
Point(3) = {5.01, 0, 0, 0.3*f};
//+
Point(4) = {15, 0, 0, f};
//+
Point(5) = {15, 3.25, 0, f};
//+
Point(6) = {0, 3.25, 0, f};
//+
Point(7) = {5, 1, 0, 0.3*f};
//+
Point(8) = {5.01, 1, 0, 0.3*f};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 1};
//+
Line(7) = {7, 2};
//+
Line(8) = {8, 7};
//+
Line(9) = {3, 8};
//+
Curve Loop(1) = {9, 8, 7, 2};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1, -7, -8, -9, 3, 4, 5, 6};
//+
Plane Surface(2) = {2};
//+
Physical Curve("inFlow", 10) = {6};
//+
Physical Curve("outFlow", 11) = {4};
//+
Physical Curve("wall", 12) = {3, 1};
//+
Physical Curve("top", 13) = {5};
//+
Physical Curve("wallSolidTop", 14) = {8};
//+
Physical Curve("wallSolidLeft", 15) = {7};
//+
Physical Curve("wallSolidRight", 16) = {9};
//+
Physical Curve("wallSolidBottom", 17) = {2};
//+
Physical Surface("solid", 18) = {1};
//+
Physical Surface("fluid", 19) = {2};
