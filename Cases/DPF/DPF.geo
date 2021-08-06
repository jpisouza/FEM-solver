// Gmsh project created on Sun Aug 01 16:16:26 2021
SetFactory("OpenCASCADE");
//+
f=0.06;
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
Line(9) = {4, 3};
//+
Line(10) = {3, 2};
//+
Line(11) = {2, 1};
//+
Line(12) = {5, 6};
//+
Line(13) = {6, 3};
//+
Line(14) = {2, 7};
//+
Line(15) = {7, 8};
//+
Line(16) = {7, 11};
//+

//+
Physical Curve("wall", 17) = {5, 4, 2, 1, 10, 14, 13};
//+
Physical Curve("in_flow", 18) = {3};
//+
Physical Curve("out_flow", 19) = {16, 7};
//+
Curve Loop(1) = {6, -7, -12, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {12, 13, -9, 4};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {9, 10, 11, 3};
//+
Plane Surface(3) = {3};
//+

//+
Curve Loop(4) = {2, -11, 14, 15};
//+
Plane Surface(4) = {4};

//+
Curve Loop(5) = {8, 1, -15, 16};
//+
Plane Surface(5) = {5};
//+
Physical Surface("free", 20) = {1, 3, 5};
//+
Physical Surface("porous", 21) = {2, 4};
//+
Physical Curve("symetry", 22) = {6, 8};
