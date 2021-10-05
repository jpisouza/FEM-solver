// Gmsh project created on Sun Aug 01 16:16:26 2021
SetFactory("OpenCASCADE");
//+
f=0.06;
Point(1) = {-4, -0.5, 0, f};
//+
Point(2) = {4, -0.5, 0, f};
//+
Point(3) = {4, 0.5, 0, f};
//+
Point(4) = {-4, 0.5, 0, f};
//+
Point(5) = {-4, 0.7, 0, f};
//+
Point(6) = {4, 0.7, 0, f};
//+
Point(7) = {4, -0.7, 0, f};
//+
Point(8) = {-4, -0.7, 0, f};
//+
Point(9) = {-4, 1.2, 0, f};
//+
Point(10) = {4, 1.2, 0, f};
//+
Point(11) = {4, -1.2, 0, f};
//+
Point(12) = {-4, -1.2, 0, f};
//+
Point(13) = {3.5, 0.7, 0, f};
//+
Point(14) = {3.5, -0.7, 0, f};
//+
Point(15) = {3.5, 0.5, 0, f};
//+
Point(16) = {3.5, -0.5, 0, f};
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
Line(9) = {4, 15};
//+
Line(10) = {3, 2};
//+
Line(11) = {1, 16};
//+
Line(12) = {5, 13};
//+
Line(13) = {6, 13};
//+
Line(14) = {14, 7};
//+
Line(15) = {14, 8};
//+
Line(16) = {7, 11};
//+
Line(17) = {13, 15};
//+
Line(18) = {16, 14};

//+
Line(19) = {15, 3};
//+
Line(20) = {16, 2};
//+
Physical Curve("in_flow", 21) = {3};
//+
Physical Curve("out_flow", 22) = {7, 13, 14, 16};
//+
Physical Curve("symmetry", 23) = {6, 8};
//+
Physical Curve("wall", 24) = {17, 4, 2, 18, 20, 10, 19, 5, 1};

//+
Curve Loop(1) = {6, -7, 13, -12, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {12, 17, -9, 4};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {3, 9, 19, 10, -20, 11};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {8, 1, -15, 14, 16};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {11, -2, -15, -18};
//+
Plane Surface(5) = {5};
//+
Physical Surface("porous", 25) = {2, 5};
//+
Physical Surface("free", 26) = {1, 3, 4};
