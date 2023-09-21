#!/usr/bin/env python

import os


files = ["lf-1000.npy",  "lf-1011.npy",  "lf-1022.npy",  "lf-1033.npy",  
         "lf-1044.npy",  "lf-1055.npy",  "lf-1066.npy",  "lf-1077.npy",
         "lf-1001.npy",  "lf-1012.npy",  "lf-1023.npy",  "lf-1034.npy",  "lf-1045.npy",  
         "lf-1056.npy",  "lf-1067.npy",  "lf-1078.npy",  "lf-1002.npy",  "lf-1013.npy",  
         "lf-1024.npy",  "lf-1035.npy",  "lf-1046.npy",  "lf-1057.npy",  "lf-1068.npy",  
         "lf-1079.npy", "lf-1003.npy",  "lf-1014.npy",  "lf-1025.npy",  "lf-1036.npy",  
         "lf-1047.npy",  "lf-1058.npy",  "lf-1069.npy",  "lf-1080.npy", "lf-1004.npy",  
         "lf-1015.npy",  "lf-1026.npy",  "lf-1037.npy",  "lf-1048.npy",  "lf-1059.npy",  
         "lf-1070.npy",  "lf-1081.npy", "lf-1005.npy",  "lf-1016.npy",  "lf-1027.npy",  
         "lf-1038.npy",  "lf-1049.npy",  "lf-1060.npy",  "lf-1071.npy",  "lf-1082.npy",
         "lf-1006.npy",  "lf-1017.npy",  "lf-1028.npy",  "lf-1039.npy",  "lf-1050.npy", 
        "lf-1061.npy",  "lf-1072.npy",  "lf-1083.npy", "lf-1007.npy",  "lf-1018.npy",  
        "lf-1029.npy",  "lf-1040.npy",  "lf-1051.npy",  "lf-1062.npy",  "lf-1073.npy",
        "lf-1008.npy",  "lf-1019.npy",  "lf-1030.npy",  "lf-1041.npy",  "lf-1052.npy", 
        "lf-1063.npy",  "lf-1074.npy", "lf-1009.npy",  "lf-1020.npy",  "lf-1031.npy", 
        "lf-1042.npy",  "lf-1053.npy",  "lf-1064.npy",  "lf-1075.npy", "lf-1010.npy", 
         "lf-1021.npy",  "lf-1032.npy",  "lf-1043.npy",  "lf-1054.npy",  "lf-1065.npy",  "lf-1076.npy"]

files_stanford = [
"lf-4100.npy",  "lf-4115.npy",  "lf-4130.npy",  "lf-4145.npy",  "lf-4160.npy",  "lf-4175.npy",  "lf-4190.npy",  "lf-4205.npy",
"lf-4101.npy",  "lf-4116.npy",  "lf-4131.npy",  "lf-4146.npy",  "lf-4161.npy",  "lf-4176.npy",  "lf-4191.npy",  "lf-4206.npy",
"lf-4102.npy",  "lf-4117.npy",  "lf-4132.npy",  "lf-4147.npy",  "lf-4162.npy",  "lf-4177.npy",  "lf-4192.npy",  "lf-4207.npy",
"lf-4103.npy",  "lf-4118.npy",  "lf-4133.npy",  "lf-4148.npy",  "lf-4163.npy",  "lf-4178.npy",  "lf-4193.npy",  "lf-4208.npy",
"lf-4104.npy",  "lf-4119.npy",  "lf-4134.npy",  "lf-4149.npy",  "lf-4164.npy",  "lf-4179.npy",  "lf-4194.npy",  "lf-4209.npy",
"lf-4105.npy",  "lf-4120.npy",  "lf-4135.npy",  "lf-4150.npy",  "lf-4165.npy",  "lf-4180.npy",  "lf-4195.npy",  "lf-4210.npy",
"lf-4106.npy",  "lf-4121.npy",  "lf-4136.npy",  "lf-4151.npy",  "lf-4166.npy",  "lf-4181.npy",  "lf-4196.npy",  "lf-4211.npy",
"lf-4107.npy",  "lf-4122.npy",  "lf-4137.npy",  "lf-4152.npy",  "lf-4167.npy",  "lf-4182.npy",  "lf-4197.npy",  "lf-4212.npy",
"lf-4108.npy",  "lf-4123.npy",  "lf-4138.npy",  "lf-4153.npy",  "lf-4168.npy",  "lf-4183.npy",  "lf-4198.npy",
"lf-4109.npy",  "lf-4124.npy",  "lf-4139.npy",  "lf-4154.npy",  "lf-4169.npy",  "lf-4184.npy",  "lf-4199.npy",
"lf-4110.npy",  "lf-4125.npy",  "lf-4140.npy",  "lf-4155.npy",  "lf-4170.npy",  "lf-4185.npy",  "lf-4200.npy",
"lf-4111.npy",  "lf-4126.npy",  "lf-4141.npy",  "lf-4156.npy",  "lf-4171.npy",  "lf-4186.npy",  "lf-4201.npy",
"lf-4112.npy",  "lf-4127.npy",  "lf-4142.npy",  "lf-4157.npy",  "lf-4172.npy",  "lf-4187.npy",  "lf-4202.npy",
"lf-4113.npy",  "lf-4128.npy",  "lf-4143.npy",  "lf-4158.npy",  "lf-4173.npy",  "lf-4188.npy",  "lf-4203.npy",
"lf-4114.npy",  "lf-4129.npy",  "lf-4144.npy",  "lf-4159.npy",  "lf-4174.npy",  "lf-4189.npy",  "lf-4204.npy",
]

with open("val_files.txt", "w+") as f:
    for file in files:
        f.write("TAMULF/test/"+file+"\n")

    for file in files_stanford:
        f.write("Stanford/test/"+file+"\n")
    