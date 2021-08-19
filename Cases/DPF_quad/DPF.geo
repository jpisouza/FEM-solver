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
Line(9) = {3, 4};
//+
Line(10) = {3, 2};
//+
Line(11) = {2, 1};
//+
Line(12) = {5, 6};
//+
Line(13) = {3, 6};
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


Physical Curve("symmetry", 22) = {6, 8};


//+
Curve Loop(1) = {9, -3, -11, -10};
//+
Plane Surface(1) = {1};

//+
Curve Loop(2) = {13, -12, -4, -9};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {7, -6, -5, 12};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {11, -2, -15, -14};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {15, -1, -8, -16};
//+
Plane Surface(5) = {5};
//+
Physical Surface("free", 23) = {3, 1, 5};
//+
Physical Surface("porous", 24) = {2, 4};
