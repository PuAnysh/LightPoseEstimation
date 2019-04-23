'''
LSP:
Right ankle
Right knee
Right hip
Left hip
Left knee
Left ankle
Right wrist
Right elbow
Right shoulder
Left shoulder
Left elbow
Left wrist
Neck
Head top
'''

'''
AI
1/右肩	
2/右肘	
3/右腕	
4/左肩	
5/左肘
6/左腕	
7/右髋	
8/右膝	
9/右踝	
10/左髋
11/左膝	
12/左踝	
13/头顶	
14/脖子
'''

LSP2AImap = {
    0:8,
    1:7,
    2:6,
    3:9,
    4:10,
    5:11,
    6:2,
    7:1,
    8:0,
    9:3,
    10:4,
    11:5,
    12:13,
    13:12
}
'''
0 - r ankle, 
1 - r knee, 
2 - r hip, 
3 - l hip, 
4 - l knee, 
5 - l ankle, 
6 - pelvis, 
7 - thorax, 
8 - upper neck, 
9 - head top, 
10 - r wrist, 
11 - r elbow, 
12 - r shoulder, 
13 - l shoulder, 
14 - l elbow, 
15 - l wrist
'''
MPII2AI = {
    0: 12,
    1: 11,
    2: 10,
    3: 13,
    4: 14,
    5: 15,
    6: 2,
    7: 1,
    8: 0,
    9: 3,
    10: 4,
    11: 5,
    12: 9,
    13: 8,
    14:-1,
    15:-1
}