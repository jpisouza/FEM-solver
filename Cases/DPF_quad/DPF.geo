// Gmsh project created on Sun Aug 01 16:16:26 2021
SetFactory("OpenCASCADE");
//+
f=1;
Point(1) = {-3, -0.5, 0, f};
//+
Point(2) = {3, -0.5, 0, f};
//+
Point(3) = {3, 0.5, 0, f};
//+
Point(4) = {-3, 0.5, 0, f};
//+
Point(5) = {-3, 0.7, 0, f};
//+
Point(6) = {3, 0.7, 0, f};
//+
Point(7) = {3, -0.7, 0, f};
//+
Point(8) = {-3, -0.7, 0, f};
//+
Point(9) = {-3, 1.2, 0, f};
//+
Point(10) = {3, 1.2, 0, f};
//+
Point(11) = {3, -1.2, 0, f};
//+
Point(12) = {-3, -1.2, 0, f};
//+
Line(1) = {12, 8};
//+
Line(2) = {8, 1};
//+
Line(3) = {1, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 9};
//+
Line(6) = {9, 10};
//+
Line(7) = {6,10};
//+
Line(8) = {11, 12};
//+
Line(9) = {3, 2};
//+
Line(10) = {2, 1};
//+
Line(11) = {6, 3};
//+
Line(12) = {2, 7};
//+
Line(13) = {7, 11};
//+
Line(14) = {5, 6};
//+
Line(15) = {4, 3};
//+
Line(16) = {2, 1};
//+
Line(17) = {7, 8};

//+
Curve Loop(1) = {6, -7, -14, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {11, -15, 4, 14};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {9, 10, 3, 15};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {12, 17, 2, -10};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {13, 8, -1, 17};
//+
Plane Surface(5) = {5};
//+
Physical Curve("in_flow", 19) = {3};
//+
Physical Curve("out_flow", 20) = {7, 13};
//+
Physical Curve("symmetry", 18) = {6, 8};
//+
Physical Curve("wall", 21) = {4, 5, 2, 1, 9, 11, 12};
//+
Physical Surface("free", 22) = {1, 3, 5};
//+
Physical Surface("porous", 23) = {2, 4};
