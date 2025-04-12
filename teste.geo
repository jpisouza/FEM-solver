// Gmsh project created on Sun Apr 06 14:26:54 2025
SetFactory("OpenCASCADE");
//+
f=0.1;
Point(1) = {1, 0, 0, f};
//+
Point(2) = {1, 1, 0, f};
//+
Point(3) = {0, 1, 0, f};
//+
Point(4) = {0, 0, 0, f};
//+
Line(1) = {3, 4};
//+
Line(2) = {4, 1};
//+
Line(3) = {1, 2};
//+
Line(4) = {2, 3};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};


//+
Physical Curve("wall", 5) = {1, 2, 3};
//+
Physical Curve("flow", 6) = {4};
//+
Physical Surface("fluid", 7) = {1};
//+

//+
Extrude {0, 0, 5} {
  Curve{4}; Surface{1}; Curve{1}; Curve{3}; Curve{2}; Layers {10}; Recombine;
}
