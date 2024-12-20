
wall = 0.08 ;

L = 1.0;


/* 
 *              5         L+3L          4
 *              o --------------------- o         
 *              |                       |       
 *         L    |                       |       
 *              |                       |       Y         
 *              o ----- o               | 2L    ^
 *              0     1 |               |       |
 *                  L   |               |       |
 *              x(0,0)  o ------------- o       o -----> X
 *                      2      3L       3
 * */

Point(0)  = {0.0,     L, 0.0,  wall}; // p0
Point(1)  = {1*L,     L, 0.0,  wall}; // p1
Point(2)  = {1*L,   0.0, 0.0,  wall}; // p2
Point(3)  = {L+6*L, 0.0, 0.0,  wall}; // p3
Point(4)  = {L+6*L, 2*L, 0.0,  wall}; // p4
Point(5)  = {0.0  , 2*L, 0.0,  wall}; // p5


Line(1) = {0, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line(5) = {4, 5};
Line(6) = {5, 0};

Physical Line('left_bound') = {6};
Physical Line('right_bound') = {4};
Physical Line('upper_bound') = {1, 2, 3, 5};
//+
Curve Loop(1) = {5, 6, 1, 2, 3, 4};
Plane Surface(1) = {1};
Physical Surface("surface") = {1};
