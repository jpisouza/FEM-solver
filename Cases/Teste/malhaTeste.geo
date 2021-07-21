// Gmsh project created on Mon Oct 05 18:24:42 2020

SetFactory("OpenCASCADE");
fator=0.2;
//+
Point(1) = {-3, -1, 0, fator};
//+
Point(2) = {3, -1, 0, fator};
//+
Point(3) = {3, 1, 0, fator};
//+
Point(4) = {-3, 1, 0, fator};
//+
Line(1) = {4, 3};
//+
Line(2) = {3, 2};
//+
Line(3) = {2, 1};
//+
Line(4) = {1, 4};
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
