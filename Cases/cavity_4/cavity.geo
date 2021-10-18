//+
fator = 0.1;
Point(1) = {0, 0, 0, fator};
//+
Point(2) = {1, 0, 0, fator};
//+
Point(3) = {1, 1, 0, fator};
//+
Point(4) = {0, 1, 0, fator};
//+
Line(1) = {4, 3};
//+
Line(2) = {3, 2};
//+
Line(3) = {2, 1};
//+
Line(4) = {1, 4};
//+
Physical Line("upper_bound") = {1};
//+
Physical Line("lower_bound") = {3};
//+
Physical Line("right_bound") = {2};
//+
Physical Line("left_bound") = {4};
//+
Line Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Surface("malha") = {1};
