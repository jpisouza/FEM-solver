// Gmsh project created on Tue Jun 22 11:18:11 2021
SetFactory("OpenCASCADE");
f = 0.05;
//+
Point(1) = {-0.75, -0.5, 0, 1.4*f};
//+
Point(2) = {1.5, -0.5, 0, 1.4*f};
//+
Point(3) = {1.5, 0.5, 0, 1.4*f};
//+
Point(4) = {-0.75, 0.5, 0, 1.4*f};
//+
Point(5) = {-0.03, 0.5, 0, 0.5*f};
//+
Point(6) = {-0.03, 0, 0, 0.5*f};
//+
Point(7) = {0.03, 0, 0, 0.5*f};
//+
Point(8) = {0.03, 0.5, 0, 0.5*f};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 8};
//+
Line(4) = {5, 4};
//+
Line(5) = {4, 1};
//+
Line(6) = {8, 5};
//+
Line(7) = {7, 8};
//+
Line(8) = {6, 7};
//+
Line(9) = {5, 6};
//+
Curve Loop(1) = {5, 1, 2, 3, 7, 8, 9, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {6, 7, 8, 9};
//+
Plane Surface(2) = {2};
//+
Physical Curve("inFlow", 10) = {5};
//+
Physical Curve("outFlow", 11) = {2};
//+
Physical Curve("wall", 12) = {1, 3, 4};
//+
Physical Curve("wallSolidBottom", 13) = {8};
//+
Physical Curve("wallSolidLeft", 14) = {9};
//+
Physical Curve("wallSolidRight", 15) = {7};
//+
Physical Curve("wallSolidTop", 16) = {6};
//+
Physical Surface("fluid", 17) = {1};
//+
Physical Surface("solid", 18) = {2};
