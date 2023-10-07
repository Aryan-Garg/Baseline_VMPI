#!/usr/bin/env python

import os


files_tam = ["lf-1000.npy",  "lf-1011.npy",  "lf-1022.npy",  "lf-1033.npy",  
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

files_hybrid = ["lf-0000.npy",  "lf-0034.npy",  "lf-0068.npy",  "lf-0102.npy",  "lf-0136.npy",  
               "lf-0170.npy",  "lf-0204.npy",  "lf-0238.npy",  "lf-0001.npy",  "lf-0035.npy",  
               "lf-0069.npy",  "lf-0103.npy",  "lf-0137.npy",  "lf-0171.npy",  "lf-0205.npy",  
               "lf-0239.npy",  "lf-0002.npy",  "lf-0036.npy",  "lf-0070.npy",  "lf-0104.npy",  
               "lf-0138.npy",  "lf-0172.npy",  "lf-0206.npy",  "lf-0240.npy",  "lf-0003.npy",  
               "lf-0037.npy",  "lf-0071.npy",  "lf-0105.npy",  "lf-0139.npy",  "lf-0173.npy",  
               "lf-0207.npy",  "lf-0241.npy",  "lf-0004.npy",  "lf-0038.npy",  "lf-0072.npy",  
               "lf-0106.npy",  "lf-0140.npy",  "lf-0174.npy",  "lf-0208.npy",  "lf-0242.npy", 
               "lf-0005.npy",  "lf-0039.npy",  "lf-0073.npy",  "lf-0107.npy",  "lf-0141.npy",  
               "lf-0175.npy",  "lf-0209.npy",  "lf-0243.npy",  "lf-0006.npy",  "lf-0040.npy",  
               "lf-0074.npy",  "lf-0108.npy",  "lf-0142.npy",  "lf-0176.npy",  "lf-0210.npy",  
               "lf-0244.npy",  "lf-0007.npy",  "lf-0041.npy",  "lf-0075.npy",  "lf-0109.npy",  
               "lf-0143.npy",  "lf-0177.npy",  "lf-0211.npy",  "lf-0245.npy",  "lf-0008.npy",  
               "lf-0042.npy",  "lf-0076.npy",  "lf-0110.npy",  "lf-0144.npy",  "lf-0178.npy",  
               "lf-0212.npy",  "lf-0246.npy",  "lf-0009.npy",  "lf-0043.npy",  "lf-0077.npy",  
               "lf-0111.npy",  "lf-0145.npy",  "lf-0179.npy",  "lf-0213.npy",  "lf-0247.npy",
               "lf-0010.npy",  "lf-0044.npy",  "lf-0078.npy",  "lf-0112.npy",  "lf-0146.npy",  
               "lf-0180.npy",  "lf-0214.npy",  "lf-0248.npy",  "lf-0011.npy",  "lf-0045.npy",  
               "lf-0079.npy",  "lf-0113.npy",  "lf-0147.npy",  "lf-0181.npy",  "lf-0215.npy",  
               "lf-0249.npy",  "lf-0012.npy",  "lf-0046.npy",  "lf-0080.npy",  "lf-0114.npy",  
               "lf-0148.npy",  "lf-0182.npy",  "lf-0216.npy",  "lf-0250.npy",  "lf-0013.npy",  
               "lf-0047.npy",  "lf-0081.npy",  "lf-0115.npy",  "lf-0149.npy",  "lf-0183.npy",  
               "lf-0217.npy",  "lf-0251.npy",  "lf-0014.npy",  "lf-0048.npy",  "lf-0082.npy",  
               "lf-0116.npy",  "lf-0150.npy",  "lf-0184.npy",  "lf-0218.npy",  "lf-0252.npy",
               "lf-0015.npy",  "lf-0049.npy",  "lf-0083.npy",  "lf-0117.npy",  "lf-0151.npy",  
               "lf-0185.npy",  "lf-0219.npy",  "lf-0253.npy",  "lf-0016.npy",  "lf-0050.npy",  
               "lf-0084.npy",  "lf-0118.npy",  "lf-0152.npy",  "lf-0186.npy",  "lf-0220.npy",  
               "lf-0254.npy",  "lf-0017.npy",  "lf-0051.npy",  "lf-0085.npy",  "lf-0119.npy",  
               "lf-0153.npy",  "lf-0187.npy",  "lf-0221.npy",  "lf-0255.npy",  "lf-0018.npy",  
               "lf-0052.npy",  "lf-0086.npy",  "lf-0120.npy",  "lf-0154.npy",  "lf-0188.npy",  
               "lf-0222.npy",  "lf-0256.npy",  "lf-0019.npy",  "lf-0053.npy",  "lf-0087.npy",  
               "lf-0121.npy",  "lf-0155.npy",  "lf-0189.npy",  "lf-0223.npy",  "lf-0257.npy",
               "lf-0020.npy",  "lf-0054.npy",  "lf-0088.npy",  "lf-0122.npy",  "lf-0156.npy",  "lf-0190.npy",  "lf-0224.npy",  "lf-0258.npy",
"lf-0021.npy",  "lf-0055.npy",  "lf-0089.npy",  "lf-0123.npy",  "lf-0157.npy",  "lf-0191.npy",  "lf-0225.npy",  "lf-0259.npy",
"lf-0022.npy",  "lf-0056.npy",  "lf-0090.npy",  "lf-0124.npy",  "lf-0158.npy",  "lf-0192.npy",  "lf-0226.npy",  "lf-0260.npy",
"lf-0023.npy",  "lf-0057.npy",  "lf-0091.npy",  "lf-0125.npy",  "lf-0159.npy",  "lf-0193.npy",  "lf-0227.npy",  "lf-0261.npy",
"lf-0024.npy",  "lf-0058.npy",  "lf-0092.npy",  "lf-0126.npy",  "lf-0160.npy",  "lf-0194.npy",  "lf-0228.npy",  "lf-0262.npy",
"lf-0025.npy",  "lf-0059.npy",  "lf-0093.npy",  "lf-0127.npy",  "lf-0161.npy",  "lf-0195.npy",  "lf-0229.npy",  "lf-0263.npy",
"lf-0026.npy",  "lf-0060.npy",  "lf-0094.npy",  "lf-0128.npy",  "lf-0162.npy",  "lf-0196.npy",  "lf-0230.npy",  "lf-0264.npy",
"lf-0027.npy",  "lf-0061.npy",  "lf-0095.npy",  "lf-0129.npy",  "lf-0163.npy",  "lf-0197.npy",  "lf-0231.npy",  "lf-0265.npy",
"lf-0028.npy",  "lf-0062.npy",  "lf-0096.npy",  "lf-0130.npy",  "lf-0164.npy",  "lf-0198.npy",  "lf-0232.npy",  "lf-0266.npy",
"lf-0029.npy",  "lf-0063.npy",  "lf-0097.npy",  "lf-0131.npy",  "lf-0165.npy",  "lf-0199.npy",  "lf-0233.npy",  "lf-0267.npy",
"lf-0030.npy",  "lf-0064.npy",  "lf-0098.npy",  "lf-0132.npy",  "lf-0166.npy",  "lf-0200.npy",  "lf-0234.npy",  "lf-0268.npy",
"lf-0031.npy",  "lf-0065.npy",  "lf-0099.npy",  "lf-0133.npy",  "lf-0167.npy",  "lf-0201.npy",  "lf-0235.npy",
"lf-0032.npy",  "lf-0066.npy",  "lf-0100.npy",  "lf-0134.npy",  "lf-0168.npy",  "lf-0202.npy",  "lf-0236.npy",
"lf-0033.npy",  "lf-0067.npy",  "lf-0101.npy",  "lf-0135.npy",  "lf-0169.npy",  "lf-0203.npy",  "lf-0237.npy"
]

kalan_test = [
"lf-0000.npy",  "lf-0004.npy",  "lf-0008.npy", "lf-0012.npy",  "lf-0016.npy",  "lf-0020.npy",  "lf-0024.npy",
"lf-0001.npy",  "lf-0005.npy",  "lf-0009.npy", "lf-0013.npy",  "lf-0017.npy",  "lf-0021.npy",
"lf-0002.npy",  "lf-0006.npy",  "lf-0010.npy", "lf-0014.npy",  "lf-0018.npy",  "lf-0022.npy",
"lf-0003.npy",  "lf-0007.npy",  "lf-0011.npy", "lf-0015.npy",  "lf-0019.npy",  "lf-0023.npy",
]

with open("TAMULF/test_files.txt", "w+") as f:
    for file in files_tam:
        f.write("TAMULF/test/"+file+"\n")

with open("Stanford/test_files.txt", "w+") as f:
    for file in files_stanford:
        f.write("Stanford/test/"+file+"\n")

with open("Hybrid/test_files.txt", "w+") as f:
    for file in files_hybrid:
        f.write("Hybrid/test/"+file+"\n")

with open("Kalantari/test_files.txt", "w+") as f:
    for file in kalan_test:
        f.write("Kalantari/test/"+file+"\n")