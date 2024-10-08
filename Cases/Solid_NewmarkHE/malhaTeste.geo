// Gmsh project created on Mon Oct 05 18:24:42 2020

SetFactory("OpenCASCADE");
fator=0.3;
//+
Point(1) = {0, 0, 0, fator};
//+
Point(2) = {10, 0, 0, fator};
//+
Point(3) = {10, 1, 0, fator};
//+
Point(4) = {0, 1, 0, fator};
//+
Line(1) = {3, 4};
//+
Line(2) = {2, 3};
//+
Line(3) = {1, 2};
//+
Line(4) = {4, 1};
//+
Line Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};

//+
Physical Line("upper_bound") = {1};
//+
Physical Line("lower_bound") = {3};
//+
Physical Line("left_bound") = {4};
//+
Physical Line("right_bound") = {2};

Physical Surface("superficie") = {1};
